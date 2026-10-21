"""Wall-clock-budgeted training loop.
Selfplay (C++ binary) -> replay buffer -> train -> timed checkpoints -> TorchScript export.
Arena vs champion when possible; champion promotion is SPRT-gated (accept H1 only on a
confirmed gain, else keep the incumbent; first run promotes unconditionally). The gate gets
a wall-clock window reserved out of the budget (eval.gate_share, default 0.1): training
stops starting iterations once the window is reached, and every gate batch is trimmed to
the deadline, so gating never pushes the run past its budget — when nothing fits, the run
publishes without gating (keeping the last 5 checkpoints) and logs that decision.
eval.ordered swaps the gate for a round-robin of the final checkpoints (equal time,
Bradley-Terry Elo) so the endgame is not decided by the max-games coin flip.
Tracks Elo vs wall-clock in log.jsonl.
Usage: python -m gai.orchestrator --config configs/competition_60m.yaml --game connect4 [--minutes 2]
       python -m gai.orchestrator --config configs/connect4_10min_test.yaml --game connect4 --dry-run
"""
import argparse, copy, glob, json, math, os, re, shutil, subprocess, time, yaml, torch
import torch.nn.functional as F
from .data import read_samples, ReplayBuffer, split_held_out
from .sprt import sprt, promote_decision, bradley_terry_elo

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p

def sh_json(cmd):
    p = run_cmd(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()[-500:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])

def build_supports_nn(build):
    p = run_cmd([f"{build}/selfplay", "--game", "connect4", "--games", "1", "--sims", "2",
                 "--out", "/tmp/gai_nn_probe.bin", "--evaluator", "nn:/tmp/nonexistent.ts"])
    return p.returncode == 0

def _eval_extra_flags(ev_cfg):
    """Extra flags for the C++ evaluate tool (equal-time evaluation plumbing).

    tc_ms     -> --tc-ms N    equal wall-clock per move for BOTH sides; appended the
                              moment the parallel C++ change lands (unknown flags are
                              only forwarded when configured, so old builds keep working).
    tc_a/tc_b -> --tc-a/--tc-b  per-side time controls (agent-type races, e.g. rollout
                              vs NN).
    All keys are read defensively so old configs keep working; values below 20 ms are
    ignored with a warning (below that a per-move budget is unusable).
    """
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    flags = []
    for key, flag in (("tc_ms", "--tc-ms"), ("tc_a", "--tc-a"), ("tc_b", "--tc-b")):
        v = _num(ev_cfg.get(key))
        if v is None:
            continue
        if v >= 20.0:
            flags += [flag, str(int(v))]
        else:
            print(f"eval.{key}={v} ignored (needs >= 20 ms)")
    return flags

def _evaluate_cmd(build, game, n, sims_a, sims_b, max_sims, a_spec, b_spec, seed, ev_cfg):
    return [f"{build}/evaluate", "--game", game, "--games", str(n),
            "--sims-a", str(sims_a), "--sims-b", str(sims_b),
            "--max-sims", str(max_sims),
            "--evaluator-a", a_spec, "--evaluator-b", b_spec,
            "--seed", str(seed)] + _eval_extra_flags(ev_cfg)

def estimate_gate_minutes_per_game(stats, ev_cfg, gate_sims):
    """Minutes per gate (evaluate) game, estimated from measured selfplay speed.

    The gate plays full games at `gate_sims` sims/move; the freshest measurement is
    the selfplay tool's games/s at stats['sims'] sims/move, scaled by the sims ratio
    and a safety factor (evaluate alternates two engines and rebuilds trees every
    game, so it is slower per game than selfplay). Without a measurement, fall back
    to a conservative constant — over-estimating only shrinks the gate, it can never
    push the run past its budget. Tune with eval.gate_time_safety (default 1.5) and
    eval.gate_minutes_per_game (default 0.25 for the no-measurement fallback).
    """
    fallback = float(ev_cfg.get("gate_minutes_per_game", 0.25))
    try:
        gps = float((stats or {}).get("games_per_s") or 0.0)
        sp_sims = float((stats or {}).get("sims") or 0.0)
    except (TypeError, ValueError):
        return fallback
    if gps > 0.0 and sp_sims > 0.0 and gate_sims > 0:
        return (1.0 / gps) * (gate_sims / sp_sims) * float(ev_cfg.get("gate_time_safety", 1.5)) / 60.0
    return fallback

def plan_gate_games(ev_cfg, gate_window_s, minutes_per_game):
    """Convert the gate's wall-clock window into a game count.

    The window is the share-limited allocation: the training loop reserves
    gate_share * budget for the endgame (it stops starting iterations once
    (1 - gate_share) * budget has elapsed), so the share is applied exactly once.
    The game count is what fits in the window:
    gate_games = min(sprt_max_games, ceil(gate_window / minutes_per_game)).
    Returns (gate_games, estimated_gate_cost_seconds); (0, 0.0) means "no room for a
    gate — publish without gating" (the caller logs that decision).
    """
    max_games = int(ev_cfg.get("sprt_max_games", 200))
    if minutes_per_game <= 0.0 or gate_window_s <= 0.0:
        return 0, 0.0
    games = max(0, min(max_games, int(math.ceil(gate_window_s / 60.0 / minutes_per_game))))
    return games, games * minutes_per_game * 60.0

def _prune_checkpoints(out_dir, keep=5):
    """Keep only the newest `keep` checkpoint_*.pt files. Returns how many existed."""
    cks = []
    for p in glob.glob(os.path.join(out_dir, "checkpoint_*.pt")):
        m = re.search(r"checkpoint_(\d+)m\.pt$", os.path.basename(p))
        cks.append(((int(m.group(1)) if m else 0), os.path.getmtime(p), p))
    cks.sort()
    for _, _, p in cks[:-keep] if keep > 0 else []:
        try:
            os.remove(p)
        except OSError:
            pass
    return len(cks)

def sprt_arena_gate(build, game, cand_spec, champ_spec, ev_cfg,
                    max_games=None, deadline_s=None, minutes_per_game=None):
    """Run candidate-vs-champion arenas in batches, early-stopping on an SPRT verdict.

    Returns (decision, stats) with decision in {'accept_h1','accept_h0','max_games'}.
    Each batch uses a fresh --seed so the opening sets are independent across batches
    (see tools/evaluate.cpp). max_games caps the wall-clock cost of the gate; callers
    pass the budget-clamped value from plan_gate_games(). deadline_s is the run's
    absolute wall-clock deadline: every batch is trimmed (or skipped) so it finishes
    before the deadline, so the gate can never push the run past its budget. A gate
    aborted for budget reports decision 'max_games' with stats['budget_aborted']=True
    (the undecided fallback in promote_decision then applies).
    """
    batch = int(ev_cfg.get("sprt_batch_games", 20))
    if max_games is None:
        max_games = int(ev_cfg.get("sprt_max_games", 200))
    sims = int(ev_cfg.get("sprt_sims", ev_cfg.get("sims", 200)))
    # Gate cost must be predictable: default to FLAT sims (no 6x headroom) so a batch
    # cannot balloon to max-sims on unstable positions. Opt back in with sprt_max_sims.
    max_sims = int(ev_cfg.get("sprt_max_sims", sims))
    elo0 = float(ev_cfg.get("sprt_elo0", 0.0))
    # elo1=35: at ~25 Elo SE per 200 games a 5-Elo hypothesis never resolves — the gate
    # always burned max_games and fell through to the W>L coin flip. 0 vs 35 resolves
    # in tens of games, which fits the gate-share budget window.
    elo1 = float(ev_cfg.get("sprt_elo1", 35.0))
    alpha = float(ev_cfg.get("sprt_alpha", 0.05))
    beta = float(ev_cfg.get("sprt_beta", 0.05))
    w = l = d = 0
    seed = 101
    dec, llr = "continue", 0.0
    resolved = False
    aborted = False
    while w + l + d < max_games:
        n = min(batch, max_games - (w + l + d))
        if deadline_s is not None and minutes_per_game and minutes_per_game > 0.0:
            fit = int((deadline_s - time.time()) / (minutes_per_game * 60.0))
            if fit <= 0:
                aborted = True
                break
            n = min(n, fit)
        res = sh_json(_evaluate_cmd(build, game, n, sims, sims, max_sims,
                                    cand_spec, champ_spec, seed, ev_cfg))
        w += res["wins"]; l += res["losses"]; d += res["draws"]
        seed += 1
        dec, llr = sprt(w, l, d, elo0, elo1, alpha, beta)
        if dec != "continue":
            resolved = True
            break
    if not resolved:
        dec = "max_games"
    return dec, dict(w=w, l=l, d=d, games=w + l + d, llr=round(llr, 3),
                     elo0=elo0, elo1=elo1, alpha=alpha, beta=beta,
                     budget_aborted=aborted)

def run_endgame_tournament(build, game, cfg, out_dir, shape, ev_cfg,
                           deadline_s=None, minutes_per_game=None, phase=None):
    """Round-robin the final checkpoints (plus the incumbent champion) at equal time
    and fit Bradley-Terry Elo from the pairwise table.

    This is the `eval.ordered` endgame: instead of a two-engine SPRT gate whose
    max_games fallback is a coin flip, every finalist plays every other finalist
    (eval.roundrobin_last checkpoints + latest + champion; equal time via eval.tc_ms
    when set), games per pair clamped to the same gate-share budget window, and the
    BT winner is promoted. A champion win keeps the incumbent. Returns None when
    infeasible (too few candidates, no budget, export failure) so the caller falls
    back to the SPRT gate / publish path.
    """
    if phase is None:
        phase = lambda msg: None
    if shape is None:
        print("endgame tournament skipped: no trained shape")
        return None
    from .model import build as build_net
    from .train import load_checkpoint, export_torchscript
    n_last = max(2, int(ev_cfg.get("roundrobin_last", 5)))
    latest_pt = os.path.join(out_dir, "latest.pt"); latest_ts = os.path.join(out_dir, "latest.ts")
    champ_pt = os.path.join(out_dir, "champion.pt"); champ_ts = os.path.join(out_dir, "champion.ts")
    cks = []
    for p in glob.glob(os.path.join(out_dir, "checkpoint_*.pt")):
        m = re.search(r"checkpoint_(\d+)m\.pt$", os.path.basename(p))
        cks.append(((int(m.group(1)) if m else 0), os.path.getmtime(p), p))
    cks.sort()
    cands = []
    for _, _, p in cks[-(n_last - 1):]:
        stem = os.path.splitext(os.path.basename(p))[0]
        cands.append((stem, os.path.join(out_dir, f"rr_{stem}.ts"), p))
    if os.path.exists(latest_pt):
        cands.append(("latest", latest_ts, latest_pt))
    if os.path.exists(champ_ts):
        cands.append(("champion", champ_ts, champ_pt))
    if len(cands) < 2:
        print(f"endgame tournament skipped: {len(cands)} candidate(s), need >= 2")
        return None
    # TorchScript-export checkpoint candidates that have no .ts yet.
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tmp_net = None
        exported = []
        for name, ts, pt in cands:
            if os.path.exists(ts):
                exported.append((name, ts, pt))
                continue
            if tmp_net is None:
                tmp_net = build_net(cfg["network"], shape["planes"], shape["h"],
                                    shape["w"], shape["actions"]).to(device)
            try:
                load_checkpoint(tmp_net, pt, device=device)
                export_torchscript(tmp_net, shape, ts)
                exported.append((name, ts, pt))
            except Exception as e:
                print(f"tournament: export failed for {name} ({e}); dropping candidate")
        cands = exported
    except Exception as e:
        print(f"endgame tournament skipped: export setup failed ({e})")
        return None
    if len(cands) < 2:
        print("endgame tournament skipped: <2 exportable candidates")
        return None
    names = [c[0] for c in cands]
    # Games per pair: the remaining gate window (already the share-limited allocation
    # reserved by the training loop) split over the pairs.
    remaining = max(0.0, (deadline_s - time.time()) if deadline_s is not None else 1e18)
    sims = int(ev_cfg.get("sims", 200))
    max_sims = int(ev_cfg.get("sprt_max_sims", sims))
    batch = int(ev_cfg.get("sprt_batch_games", 20))
    n_pairs = len(names) * (len(names) - 1) // 2
    if minutes_per_game and minutes_per_game > 0.0:
        affordable_total = int(remaining / 60.0 / minutes_per_game)
    else:
        affordable_total = int(ev_cfg.get("roundrobin_games", 20)) * n_pairs
    gpp = min(int(ev_cfg.get("roundrobin_games", 20)),
              affordable_total // n_pairs if n_pairs else 0)
    if gpp < 2:
        print(f"endgame tournament skipped: budget fits only {gpp} games/pair "
              f"({n_pairs} pairs, {remaining:.0f}s remaining)")
        return None
    est_s = n_pairs * gpp * (minutes_per_game or 0.0) * 60.0
    phase(f"endgame tournament: {len(names)} candidates, {n_pairs} pairs x {gpp} games (~{est_s:.0f}s)")
    flags = _eval_extra_flags(ev_cfg)
    pair_results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w = l = d = 0
            played = 0
            seed = 1000 + i * 100 + j
            while played < gpp:
                n = min(batch, gpp - played)
                if (deadline_s is not None and minutes_per_game and minutes_per_game > 0.0):
                    fit = int((deadline_s - time.time()) / (minutes_per_game * 60.0))
                    if fit <= 0:
                        print(f"tournament: out of budget before {names[i]} vs {names[j]} batch")
                        break
                    n = min(n, fit)
                try:
                    res = sh_json(_evaluate_cmd(build, game, n, sims, sims, max_sims,
                                                f"nn:{cands[i][1]}", f"nn:{cands[j][1]}",
                                                seed, ev_cfg))
                except Exception as e:
                    print(f"tournament: pair {names[i]} vs {names[j]} failed ({e})")
                    break
                w += res["wins"]; l += res["losses"]; d += res["draws"]
                played += n; seed += 1
            if played > 0:
                pair_results[(i, j)] = (w, l, d)
                print(f"tournament: {names[i]} vs {names[j]}: {w}-{l}-{d}")
    if not pair_results:
        print("endgame tournament produced no pairs")
        return None
    elo = bradley_terry_elo(names, pair_results)
    # Tie-break by candidate recency; "champion" is last in `cands`, so an equal-elo
    # incumbent keeps the throne.
    winner = max(names, key=lambda n: (elo[n], names.index(n)))
    return dict(names=names, candidates=cands,
                ts_of={c[0]: c[1] for c in cands}, pt_of={c[0]: c[2] for c in cands},
                pairs=pair_results, elo=elo, winner=winner, games_per_pair=gpp)

def dry_run_override(cfg):
    """Compress a config into a ~2-minute validation run (in-memory only).

    2 iters x 1 train step, tiny selfplay, TorchScript export, 2-game arena, no gate.
    Never requires a GPU: rollout evaluator, AMP off, NN switch disabled by pushing
    nn_start_iter past the iteration cap.
    """
    cfg = copy.deepcopy(cfg)
    cfg["budget_minutes"] = 2
    cfg.setdefault("checkpoint", {})["interval_minutes"] = 0.05  # arena fires after iter 1
    sp = cfg.setdefault("selfplay", {})
    sp["games_per_iter"] = min(int(sp.get("games_per_iter", 4)), 4)
    sp["simulations"] = min(int(sp.get("simulations", 50)), 50)
    sp["evaluator"] = "rollout"
    sp["nn_start_iter"] = 10 ** 9  # never switch to NN inside the dry run
    sp.pop("hybrid", None)
    tr = cfg.setdefault("training", {})
    tr["steps_per_iter"] = 1
    tr["batch_size"] = min(int(tr.get("batch_size", 32)), 32)
    tr["replay_capacity"] = min(int(tr.get("replay_capacity", 8192)), 8192)
    tr["mixed_precision"] = False
    tr.pop("lr_schedule", None)
    tr.pop("anneal_frac", None)
    ev = cfg.setdefault("eval", {})
    ev["games"] = 2
    ev["sims"] = 50
    ev["sprt"] = False
    ev["ordered"] = False
    return cfg

def dry_run_plan(cfg, dry_iters=2):
    sp, tr, ev = cfg["selfplay"], cfg["training"], cfg.get("eval", {})
    print("=== dry-run plan (in-memory overrides; the config file is untouched) ===")
    print(f"  budget: {cfg['budget_minutes']} min (cap {dry_iters} iters), "
          f"train steps/iter: {tr['steps_per_iter']}, batch {tr['batch_size']}")
    print(f"  t=0     probe build nn support")
    print(f"  iter i: selfplay {sp['games_per_iter']} games @ {sp['simulations']} sims "
          f"({sp.get('evaluator', 'rollout')}) -> holdout split -> train -> export latest.ts")
    print(f"  every {cfg['checkpoint']['interval_minutes']} min: checkpoint + arena "
          f"({ev.get('games', 20)} eval games @ {ev.get('sims', 200)} sims)")
    print(f"  end:    publish champion (SPRT gate {'on' if ev.get('sprt', True) else 'off'}, "
          f"ordered tournament {'on' if ev.get('ordered', False) else 'off'})")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--game", required=True)
    ap.add_argument("--build", default="build"); ap.add_argument("--minutes", type=float); ap.add_argument("--out", default="models")
    ap.add_argument("--dry-run", action="store_true",
                    help="compressed ~2-min validation run (in-memory config override: "
                         "2 iters, 1 train step, export, 2 eval games, no gate); "
                         "requires the C++ build but never a GPU")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); os.makedirs(a.out, exist_ok=True)
    dry = bool(a.dry_run)
    if dry:
        cfg = dry_run_override(cfg)
        dry_run_plan(cfg)
    budget = (a.minutes if a.minutes else cfg["budget_minutes"]) * 60; ck_every = cfg["checkpoint"]["interval_minutes"] * 60
    # Reserve the endgame gate window out of the budget (eval.gate_share, default 0.1):
    # the training loop stops starting iterations once the training window is spent, so
    # the promotion gate/tournament always has wall-clock left and runs inside the
    # budget instead of overrunning it. With both sprt and ordered disabled nothing is
    # reserved (legacy behaviour). LR schedules span the training window, not the total.
    ev0 = cfg.get("eval", {}) or {}
    endgame_enabled = bool(ev0.get("sprt", True) or ev0.get("ordered", False))
    gate_reserve_s = budget * float(ev0.get("gate_share", 0.1)) if endgame_enabled else 0.0
    train_budget = budget - gate_reserve_s
    from .train import make_trainer, save_checkpoint, export_torchscript
    t0 = time.time()
    def phase(msg):
        if dry:
            print(f"[dry-run t=+{time.time() - t0:.1f}s] {msg}")
    dry_max_iters = 2 if dry else None
    nn_ok = build_supports_nn(a.build)
    print(f"build nn support: {nn_ok}")
    phase(f"build nn support: {nn_ok}")
    next_ck = ck_every; buf = None; net = opt = step = None
    log = open(os.path.join(a.out, "log.jsonl"), "a"); it = 0
    shape = None
    champ_ts = None
    last_stats = None
    while time.time() - t0 < train_budget and (dry_max_iters is None or it < dry_max_iters):
        data = os.path.join(a.out, "selfplay_tmp.bin"); sp = cfg["selfplay"]
        latest_ts = os.path.join(a.out, "latest.ts")
        ev = sp.get("evaluator", "rollout")
        # Gated + annealed evaluator switch: below nn_start_iter always rollout
        # (early nets are worse than rollout and poison data); then ramp
        # hybrid alpha 0 -> 1 over nn_anneal_iters; alpha=1 == pure NN.
        nn_start = sp.get("nn_start_iter", 5)
        nn_anneal = max(1, sp.get("nn_anneal_iters", 40))
        alpha = 0.0
        if it < nn_start:
            ev = "rollout"
        elif nn_ok and os.path.exists(latest_ts) and it > 0:
            tag = "nn-strict" if sp.get("nn_strict", 0) else "nn"
            if sp.get("hybrid", 0):
                alpha = min(1.0, (it - nn_start + 1) / nn_anneal)
                ev = f"hybrid:{latest_ts}:{alpha:.3f}"
            else:
                alpha = 1.0
                ev = f"{tag}:{latest_ts}"
        # playout-cap randomization flags come from config when present
        cmd = [f"{a.build}/selfplay", "--game", a.game, "--games", str(sp["games_per_iter"]), "--sims", str(sp["simulations"]),
               "--out", data, "--evaluator", ev, "--seed", str(it + 1),
               "--full-prob", str(sp.get("full_prob", 1.0)), "--cheap-frac", str(sp.get("cheap_frac", 0.125)),
               "--forced", str(sp.get("forced", 1)), "--prune", str(sp.get("prune", 1)),
               "--shortcuts", str(sp.get("shortcuts", 1)),
               "--gumbel", str(sp.get("gumbel", 0)), "--gumbel-sims", str(sp.get("gumbel_sims", 32)),
               "--value-lambda", str(sp.get("value_lambda", 0.1)),
               "--prior-temp", str(sp.get("prior_temp", 1.0)),
               "--resign-q", str(sp.get("resign_q", -2.0)),
               "--resign-min-plies", str(sp.get("resign_min_plies", 10)),
               "--resign-probe", str(sp.get("resign_probe", 0.0)),
               "--batch", str(sp.get("batch", 1)),
               "--dirichlet", str(sp.get("dirichlet", 0.3)),
               "--temp-moves", str(sp.get("temperature_moves", 10)),
               "--loss-fallthrough", str(sp.get("loss_fallthrough", 1)),
               "--gumbel-sigma-mctx", str(sp.get("gumbel_sigma_mctx", 0)),
               "--lanes", str(sp.get("lanes", 1))]
        # threads "auto" (or absent) = let the tool default to hardware_concurrency;
        # atol("auto")==0 would spawn zero workers, so only forward numeric values.
        # NOTE: fallback path below must mirror this (thread-count drift = oversubscription).
        def _maybe_add_threads(cmd):
            try:
                int(sp.get("threads", "auto"))
                cmd += ["--threads", str(sp["threads"])]
            except (ValueError, TypeError, KeyError):
                pass
            return cmd
        try:
            int(sp.get("threads", "auto"))
            cmd += ["--threads", str(sp["threads"])]
        except (ValueError, TypeError, KeyError):
            pass
        try:
            stats = sh_json(cmd)
        except RuntimeError as e:
            if ev.startswith("nn-strict:"):
                # Strict mode: a failing NN evaluator must kill the run, never
                # silently fall back to rollout data.
                raise RuntimeError(f"nn-strict evaluator failed, aborting: {e}")
            if ev.startswith("nn:") or ev.startswith("hybrid:"):  # fall back to CPU evaluator
                nn_ok = False
                print(f"nn evaluator failed, falling back: {e}")
                ev = sp.get("evaluator", "rollout")
                fb = [f"{a.build}/selfplay", "--game", a.game, "--games", str(sp["games_per_iter"]),
                                 "--sims", str(sp["simulations"]), "--out", data, "--evaluator", ev, "--seed", str(it + 1),
                                 "--full-prob", str(sp.get("full_prob", 1.0)), "--cheap-frac", str(sp.get("cheap_frac", 0.125)),
                                 "--forced", str(sp.get("forced", 1)), "--prune", str(sp.get("prune", 1)),
                                 "--shortcuts", str(sp.get("shortcuts", 1)),
                                 "--gumbel", str(sp.get("gumbel", 0)), "--gumbel-sims", str(sp.get("gumbel_sims", 32)),
                                 "--value-lambda", str(sp.get("value_lambda", 0.1)),
                                 "--prior-temp", str(sp.get("prior_temp", 1.0)),
                                 "--resign-q", str(sp.get("resign_q", -2.0)),
                                 "--resign-min-plies", str(sp.get("resign_min_plies", 10)),
                                 "--resign-probe", str(sp.get("resign_probe", 0.0)),
                                 "--batch", str(sp.get("batch", 1)),
                                 "--dirichlet", str(sp.get("dirichlet", 0.3)),
               "--temp-moves", str(sp.get("temperature_moves", 10)),
                                 "--loss-fallthrough", str(sp.get("loss_fallthrough", 1)),
                                 "--gumbel-sigma-mctx", str(sp.get("gumbel_sigma_mctx", 0)),
               "--lanes", str(sp.get("lanes", 1))]
                fb = _maybe_add_threads(fb)
                stats = sh_json(fb)
            else:
                raise
        last_stats = stats
        phase(f"iter {it}: selfplay {stats.get('games')} games @ {stats.get('sims')} sims "
              f"in {stats.get('seconds', '?')}s ({stats.get('games_per_s', 0):.2f} games/s)")
        x, pi, z, shape = read_samples(data)
        full = shape.get("full")
        # Hold-out split BEFORE augmentation: augmenting first makes held-out
        # positions mirror twins of training positions (audit #8). Split raw,
        # then augment the train portion only.
        # holdout_frac is configurable (audit C1): the held-out split is thrown
        # away from training every iteration, so 10% of self-play data was
        # burned by default. Default stays 0.1 for backward compatibility;
        # competition configs set 0.02.
        (train_x, train_pi, train_z, train_full,
         holdout_x, holdout_pi, holdout_z, holdout_full) = split_held_out(
             x, pi, z, full, holdout_frac=cfg["training"].get("holdout_frac", 0.1))
        if sp.get("augment", False):
            from .data import maybe_mirror_batch
            train_x, train_pi, train_z, train_full = maybe_mirror_batch(
                train_x, train_pi, train_z, train_full)
        if buf is None:
            buf = ReplayBuffer(cfg["training"]["replay_capacity"], shape["planes"], shape["h"], shape["w"], shape["actions"])
            tr = cfg["training"]
            net, opt, step, set_lr = make_trainer(
                cfg["network"], shape, tr["lr"], tr.get("weight_decay", 1e-4),
                tr.get("mixed_precision", True),
                value_weight=tr.get("value_weight", 1.0),
                grad_clip=tr.get("grad_clip", 0.0),
                optimizer=tr.get("optimizer", "adamw"),
                momentum=tr.get("momentum", 0.9))
            base_lr = tr["lr"]
            annealed = False
        # (split already done above, before augmentation)
        # Evaluate network on held-out data for monitoring.
        # Audit C1: the policy metric uses full-search samples only -- the
        # training loss masks cheap-search positions (policy_pruning), so
        # including them made the monitor read high for reasons unrelated
        # to progress.
        net.eval()
        device = next(net.parameters()).device
        with torch.no_grad():
            holdout_full_mask = torch.as_tensor(holdout_full == 1.0, device=device)
            if int(holdout_full_mask.sum().item()) > 0:
                holdout_x_t = torch.as_tensor(holdout_x, device=device)[holdout_full_mask]
                holdout_pi_t = torch.as_tensor(holdout_pi, device=device)[holdout_full_mask]
                holdout_z_t = torch.as_tensor(holdout_z, device=device)[holdout_full_mask]
                logits_ho, v_ho = net(holdout_x_t)
                # Policy cross-entropy (average over held-out samples, uniform weights)
                per = -(holdout_pi_t * F.log_softmax(logits_ho.float(), -1)).sum(-1)
                heldout_policy_loss = per.mean().item()
                # Value MSE
                heldout_value_mse = F.mse_loss(v_ho.squeeze(), holdout_z_t).item()
            else:
                heldout_policy_loss = float("nan")
                heldout_value_mse = float("nan")
        # Add only train portion to replay buffer
        buf.add(train_x, train_pi, train_z, train_full)
        # Update-to-data ratio control: clamp steps_per_iter so
        # (steps * batch) / train_positions_added <= max_reuse_ratio, where the
        # denominator is the actual post-augment train positions added to the
        # buffer this iter (raw len(z) would undercount ~2x and starve training).
        max_reuse_ratio = cfg["training"].get("max_reuse_ratio", 3.0)
        fresh_positions = len(train_z)
        batch_size = cfg["training"]["batch_size"]
        max_allowed_steps = int(max_reuse_ratio * fresh_positions / batch_size)
        # Guard: tiny batches (e.g. early iters) must still train at least 1 step
        # when the buffer holds a full batch; 0-step iters silently stall learning.
        if max_allowed_steps <= 0 and buf.n >= batch_size:
            max_allowed_steps = 1
        effective_steps = min(cfg["training"]["steps_per_iter"], max_allowed_steps)
        # LR schedule: cosine decay over the training window (default when
        # lr_schedule: cosine) — training ends at budget - gate_reserve, so the decay
        # must bottom out there, not at the full budget. Else legacy one-shot drop in
        # last anneal_frac. Cosine wins for final-checkpoint Elo.
        el_pre = time.time() - t0
        tr = cfg["training"]
        if tr.get("lr_schedule", "") == "cosine":
            frac = min(1.0, el_pre / train_budget)
            warm = tr.get("warmup_frac", 0.03)
            if frac < warm and warm > 0:
                lr_now = base_lr * (frac / warm)
            else:
                lr_now = base_lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            set_lr(lr_now)
            if int(el_pre) % 600 < 12:
                print(f"cosine lr {lr_now:.6f} at el={el_pre:.0f}s frac={frac:.2f}")
        elif (tr.get("anneal_frac", 0) > 0 and not annealed
                and el_pre > train_budget * (1.0 - tr["anneal_frac"])):
            set_lr(base_lr * tr.get("anneal_factor", 0.1))
            annealed = True
            print(f"anneal: lr {base_lr} -> {base_lr * tr.get('anneal_factor', 0.1)} at el={el_pre:.0f}s")
        # Train for effective_steps
        rec_w = tr.get("recency_weight", 0.0)
        losses = {}
        for _ in range(effective_steps):
            try:
                losses = step(*buf.sample(batch_size, recency=rec_w))
            except TypeError:
                xb, pib, zb = buf.sample(batch_size, recency=rec_w)[:3]
                losses = step(xb, pib, zb)
        el = time.time() - t0
        phase(f"iter {it}: trained {effective_steps} step(s)")
        log.write(json.dumps(dict(iter=it, elapsed_s=round(el, 1), selfplay=stats,
                                replay=buf.n,
                                heldout_policy=heldout_policy_loss,
                                heldout_value=heldout_value_mse,
                                effective_steps=effective_steps,
                                max_reuse_ratio=max_reuse_ratio,
                                recency_weight=rec_w,
                                fresh_positions=fresh_positions,
                                evaluator=ev, alpha=round(alpha, 3),
                                **losses)) + "\n")
        log.flush(); it += 1
        save_checkpoint(net, os.path.join(a.out, "latest.pt"))
        try:
            export_torchscript(net, shape, latest_ts)
        except Exception as e:
            # A failed export leaves a STALE latest.ts in use while the log claims
            # evaluator=nn:..., silently poisoning selfplay. Fatal by default;
            # set training.allow_stale_export to keep the old warn-and-continue.
            if not cfg["training"].get("allow_stale_export", False):
                raise RuntimeError(f"torchscript export failed, aborting: {e}")
            print(f"export ts failed (continuing with stale {latest_ts}): {e}")
        phase(f"iter {it - 1}: checkpoint + exported {os.path.basename(latest_ts)}")
        if el >= next_ck:
            ck = os.path.join(a.out, f"checkpoint_{int(next_ck // 60)}m.pt")
            save_checkpoint(net, ck); next_ck += ck_every
            # Arena that actually measures the net: NN (latest.ts) vs rollout baseline.
            # Old code compared rollout-vs-rollout (identical 16W-4L every run) so Elo never moved.
            try:
                ev_cfg = cfg.get("eval", {})
                arena_cmd = [f"{a.build}/evaluate", "--game", a.game, "--games", str(ev_cfg.get("games", 20)),
                                 "--sims-a", str(ev_cfg.get("sims", 200)), "--sims-b", "50"] + _eval_extra_flags(ev_cfg)
                if os.path.exists(latest_ts):
                    arena_cmd += ["--evaluator-a", f"nn:{latest_ts}", "--evaluator-b", "rollout"]
                arena = sh_json(arena_cmd)
                log.write(json.dumps(dict(iter=it, elapsed_s=round(time.time() - t0, 1), arena_vs_weak=arena, checkpoint=ck)) + "\n"); log.flush()
                phase(f"checkpoint arena: {arena}")
            except Exception as e:
                print(f"arena failed: {e}")
    save_checkpoint(net, os.path.join(a.out, "latest.pt"))
    # SPRT-gated champion promotion: only replace the champion when latest.ts is
    # confirmed stronger by the sequential test; otherwise keep the incumbent.
    # First run (no champion.ts yet) promotes unconditionally.
    latest_pt = os.path.join(a.out, "latest.pt")
    latest_ts_f = os.path.join(a.out, "latest.ts")
    champ_pt = os.path.join(a.out, "champion.pt")
    champ_ts_f = os.path.join(a.out, "champion.ts")
    ev_cfg = cfg.get("eval", {}) or {}
    # The gate runs strictly inside the wall-clock budget: the training loop reserved
    # gate_share * budget for the endgame, so remaining_s is the gate window. Convert
    # it to a game count and skip gating entirely when nothing fits (publish instead).
    remaining_s = max(0.0, budget - (time.time() - t0))
    gate_sims = int(ev_cfg.get("sprt_sims", ev_cfg.get("sims", 200)))
    mpg = estimate_gate_minutes_per_game(last_stats, ev_cfg, gate_sims)
    gate_games, gate_cost_s = plan_gate_games(ev_cfg, remaining_s, mpg)
    phase(f"endgame: {remaining_s:.0f}s window, gate plan {gate_games} games "
          f"(~{gate_cost_s:.0f}s at {mpg:.3f} min/game)")
    promote, gate, published = True, None, False
    tournament = None
    if ev_cfg.get("ordered", False):
        tournament = run_endgame_tournament(
            a.build, a.game, cfg, a.out, shape, ev_cfg,
            deadline_s=t0 + budget, minutes_per_game=mpg, phase=phase)
    if tournament is not None:
        # Ordered endgame: the Bradley-Terry winner over the final checkpoints is the
        # promotion decision (a champion win keeps the incumbent). This replaces the
        # max-games W>L coin flip for the endgame.
        winner = tournament["winner"]
        promote = winner != "champion"
        gate = dict(decision="ordered_tournament", winner=winner,
                    elo={k: round(v, 1) for k, v in tournament["elo"].items()},
                    pairs={f"{tournament['names'][i]}|{tournament['names'][j]}": list(v)
                           for (i, j), v in tournament["pairs"].items()},
                    names=tournament["names"],
                    games_per_pair=tournament["games_per_pair"])
        print(f"endgame tournament: winner={winner} elo={gate['elo']} -> promote={promote}")
        if promote:
            try:
                shutil.copy(tournament["pt_of"][winner], champ_pt)
                shutil.copy(tournament["ts_of"][winner], champ_ts_f)
                published = True
            except Exception as e:
                print(f"champion promote (tournament) failed: {e}; publishing latest")
    elif ev_cfg.get("sprt", True) and os.path.exists(champ_ts_f) and os.path.exists(latest_ts_f):
        if remaining_s <= 0.0 or gate_games <= 0:
            reason = (f"insufficient budget: {remaining_s:.0f}s left of the "
                      f"{gate_reserve_s:.0f}s gate window fits {gate_games} games "
                      f"(at {mpg:.3f} min/game)")
            print(f"SPRT gate skipped ({reason}); publishing latest, keeping last 5 checkpoints")
            kept = _prune_checkpoints(a.out, keep=5)
            gate = dict(decision="skipped_no_budget", reason=reason,
                        remaining_s=round(remaining_s, 1),
                        estimated_gate_cost_s=round(gate_cost_s, 1),
                        minutes_per_gate_game=round(mpg, 4),
                        checkpoints_kept=kept)
        else:
            try:
                dec, st = sprt_arena_gate(a.build, a.game, f"nn:{latest_ts_f}", f"nn:{champ_ts_f}",
                                          ev_cfg, max_games=gate_games,
                                          deadline_s=t0 + budget, minutes_per_game=mpg)
                gate = dict(decision=dec, **st)
                # Promote on a confirmed gain; a max-games non-decision falls back to W>L.
                promote = promote_decision(dec, st)
                print(f"SPRT champion gate: {dec} {st} -> promote={promote}")
            except Exception as e:
                print(f"SPRT gate failed ({e}); promoting latest")
    try:
        if promote and not published and os.path.exists(latest_pt):
            shutil.copy(latest_pt, champ_pt)
            if nn_ok and shape is not None:
                export_torchscript(net, shape, champ_ts_f)
    except Exception as e:
        print(f"champion promote failed: {e}")
    if gate is not None:
        log.write(json.dumps(dict(iter=it, elapsed_s=round(time.time() - t0, 1),
                                  sprt_gate=gate, promoted=promote)) + "\n"); log.flush()
    phase("done")

if __name__ == "__main__": main()
