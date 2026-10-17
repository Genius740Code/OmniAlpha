"""Wall-clock-budgeted training loop.
Selfplay (C++ binary) -> replay buffer -> train -> timed checkpoints -> TorchScript export.
Arena vs champion when possible; champion promotion is SPRT-gated (accept H1 only on a
confirmed gain, else keep the incumbent; first run promotes unconditionally).
Tracks Elo vs wall-clock in log.jsonl.
Usage: python -m gai.orchestrator --config configs/competition_60m.yaml --game connect4 [--minutes 2]"""
import argparse, json, math, os, subprocess, time, yaml, torch
import torch.nn.functional as F
from .data import read_samples, ReplayBuffer, split_held_out
from .sprt import sprt, promote_decision

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

def sprt_arena_gate(build, game, cand_spec, champ_spec, ev_cfg):
    """Run candidate-vs-champion arenas in batches, early-stopping on an SPRT verdict.

    Returns (decision, stats) with decision in {'accept_h1','accept_h0','max_games'}.
    Each batch uses a fresh --seed so the opening sets are independent across batches
    (see tools/evaluate.cpp). max_games caps the wall-clock cost of the gate.
    """
    batch = int(ev_cfg.get("sprt_batch_games", 20))
    max_games = int(ev_cfg.get("sprt_max_games", 200))
    sims = int(ev_cfg.get("sprt_sims", ev_cfg.get("sims", 200)))
    # Gate cost must be predictable: default to FLAT sims (no 6x headroom) so a batch
    # cannot balloon to max-sims on unstable positions. Opt back in with sprt_max_sims.
    max_sims = int(ev_cfg.get("sprt_max_sims", sims))
    elo0 = float(ev_cfg.get("sprt_elo0", 0.0))
    elo1 = float(ev_cfg.get("sprt_elo1", 5.0))
    alpha = float(ev_cfg.get("sprt_alpha", 0.05))
    beta = float(ev_cfg.get("sprt_beta", 0.05))
    w = l = d = 0
    seed = 101
    dec, llr = "continue", 0.0
    while w + l + d < max_games:
        n = min(batch, max_games - (w + l + d))
        res = sh_json([f"{build}/evaluate", "--game", game, "--games", str(n),
                       "--sims-a", str(sims), "--sims-b", str(sims),
                       "--max-sims", str(max_sims),
                       "--evaluator-a", cand_spec, "--evaluator-b", champ_spec,
                       "--seed", str(seed)])
        w += res["wins"]; l += res["losses"]; d += res["draws"]
        seed += 1
        dec, llr = sprt(w, l, d, elo0, elo1, alpha, beta)
        if dec != "continue":
            break
    else:
        dec = "max_games"
    return dec, dict(w=w, l=l, d=d, games=w + l + d, llr=round(llr, 3),
                     elo0=elo0, elo1=elo1, alpha=alpha, beta=beta)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--game", required=True)
    ap.add_argument("--build", default="build"); ap.add_argument("--minutes", type=float); ap.add_argument("--out", default="models")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); os.makedirs(a.out, exist_ok=True)
    budget = (a.minutes if a.minutes else cfg["budget_minutes"]) * 60; ck_every = cfg["checkpoint"]["interval_minutes"] * 60
    from .train import make_trainer, save_checkpoint, export_torchscript
    nn_ok = build_supports_nn(a.build)
    print(f"build nn support: {nn_ok}")
    t0 = time.time(); next_ck = ck_every; buf = None; net = opt = step = None
    log = open(os.path.join(a.out, "log.jsonl"), "a"); it = 0
    shape = None
    champ_ts = None
    while time.time() - t0 < budget:
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
        # LR schedule: cosine decay over wall-clock budget (default when lr_schedule: cosine),
        # else legacy one-shot drop in last anneal_frac. Cosine wins for final-checkpoint Elo.
        el_pre = time.time() - t0
        tr = cfg["training"]
        if tr.get("lr_schedule", "") == "cosine":
            frac = min(1.0, el_pre / budget)
            warm = tr.get("warmup_frac", 0.03)
            if frac < warm and warm > 0:
                lr_now = base_lr * (frac / warm)
            else:
                lr_now = base_lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * frac)))
            set_lr(lr_now)
            if int(el_pre) % 600 < 12:
                print(f"cosine lr {lr_now:.6f} at el={el_pre:.0f}s frac={frac:.2f}")
        elif (tr.get("anneal_frac", 0) > 0 and not annealed
                and el_pre > budget * (1.0 - tr["anneal_frac"])):
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
        if el >= next_ck:
            ck = os.path.join(a.out, f"checkpoint_{int(next_ck // 60)}m.pt")
            save_checkpoint(net, ck); next_ck += ck_every
            # Arena that actually measures the net: NN (latest.ts) vs rollout baseline.
            # Old code compared rollout-vs-rollout (identical 16W-4L every run) so Elo never moved.
            try:
                ev_cfg = cfg.get("eval", {})
                arena_cmd = [f"{a.build}/evaluate", "--game", a.game, "--games", str(ev_cfg.get("games", 20)),
                                 "--sims-a", str(ev_cfg.get("sims", 200)), "--sims-b", "50"]
                if os.path.exists(latest_ts):
                    arena_cmd += ["--evaluator-a", f"nn:{latest_ts}", "--evaluator-b", "rollout"]
                arena = sh_json(arena_cmd)
                log.write(json.dumps(dict(iter=it, elapsed_s=round(time.time() - t0, 1), arena_vs_weak=arena, checkpoint=ck)) + "\n"); log.flush()
            except Exception as e:
                print(f"arena failed: {e}")
    save_checkpoint(net, os.path.join(a.out, "latest.pt"))
    # SPRT-gated champion promotion: only replace the champion when latest.ts is
    # confirmed stronger by the sequential test; otherwise keep the incumbent.
    # First run (no champion.ts yet) promotes unconditionally.
    import shutil
    latest_pt = os.path.join(a.out, "latest.pt")
    latest_ts_f = os.path.join(a.out, "latest.ts")
    champ_pt = os.path.join(a.out, "champion.pt")
    champ_ts_f = os.path.join(a.out, "champion.ts")
    ev_cfg = cfg.get("eval", {})
    promote, gate = True, None
    if ev_cfg.get("sprt", True) and os.path.exists(champ_ts_f) and os.path.exists(latest_ts_f):
        try:
            dec, st = sprt_arena_gate(a.build, a.game, f"nn:{latest_ts_f}", f"nn:{champ_ts_f}", ev_cfg)
            gate = dict(decision=dec, **st)
            # Promote on a confirmed gain; a max-games non-decision falls back to W>L.
            promote = promote_decision(dec, st)
            print(f"SPRT champion gate: {dec} {st} -> promote={promote}")
        except Exception as e:
            print(f"SPRT gate failed ({e}); promoting latest")
    try:
        if promote:
            shutil.copy(latest_pt, champ_pt)
            if nn_ok and shape is not None:
                export_torchscript(net, shape, champ_ts_f)
    except Exception as e:
        print(f"champion promote failed: {e}")
    if gate is not None:
        log.write(json.dumps(dict(iter=it, elapsed_s=round(time.time() - t0, 1),
                                  sprt_gate=gate, promoted=promote)) + "\n"); log.flush()

if __name__ == "__main__": main()
