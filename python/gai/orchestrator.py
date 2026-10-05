"""Wall-clock-budgeted training loop.
Selfplay (C++ binary) -> replay buffer -> train -> timed checkpoints -> TorchScript export.
Arena vs champion when possible; otherwise promotes latest. Tracks Elo vs wall-clock in log.jsonl.
Usage: python -m gai.orchestrator --config configs/competition_60m.yaml --game connect4 [--minutes 2]"""
import argparse, json, os, subprocess, time, yaml, torch
import torch.nn.functional as F
from .data import read_samples, ReplayBuffer, split_held_out

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
            if sp.get("hybrid", 0):
                alpha = min(1.0, (it - nn_start + 1) / nn_anneal)
                ev = f"hybrid:{latest_ts}:{alpha:.3f}"
            else:
                alpha = 1.0
                ev = f"nn:{latest_ts}"
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
               "--batch", str(sp.get("batch", 1)),
               "--dirichlet", str(sp.get("dirichlet", 0.3)),
               "--loss-fallthrough", str(sp.get("loss_fallthrough", 1)),
               "--gumbel-sigma-mctx", str(sp.get("gumbel_sigma_mctx", 0))]
        # threads "auto" (or absent) = let the tool default to hardware_concurrency;
        # atol("auto")==0 would spawn zero workers, so only forward numeric values.
        try:
            int(sp.get("threads", "auto"))
            cmd += ["--threads", str(sp["threads"])]
        except (ValueError, TypeError, KeyError):
            pass
        try:
            stats = sh_json(cmd)
        except RuntimeError as e:
            if ev.startswith("nn:") or ev.startswith("hybrid:"):  # fall back to CPU evaluator
                nn_ok = False
                print(f"nn evaluator failed, falling back: {e}")
                ev = sp.get("evaluator", "rollout")
                stats = sh_json([f"{a.build}/selfplay", "--game", a.game, "--games", str(sp["games_per_iter"]),
                                 "--sims", str(sp["simulations"]), "--out", data, "--evaluator", ev, "--seed", str(it + 1),
                                 "--full-prob", str(sp.get("full_prob", 1.0)), "--cheap-frac", str(sp.get("cheap_frac", 0.125)),
                                 "--forced", str(sp.get("forced", 1)), "--prune", str(sp.get("prune", 1)),
                                 "--shortcuts", str(sp.get("shortcuts", 1)),
                                 "--gumbel", str(sp.get("gumbel", 0)), "--gumbel-sims", str(sp.get("gumbel_sims", 32)),
                                 "--value-lambda", str(sp.get("value_lambda", 0.1)),
                                 "--prior-temp", str(sp.get("prior_temp", 1.0)),
                                 "--resign-q", str(sp.get("resign_q", -2.0)),
                                 "--resign-min-plies", str(sp.get("resign_min_plies", 10)),
                                 "--batch", str(sp.get("batch", 1)),
                                 "--dirichlet", str(sp.get("dirichlet", 0.3)),
                                 "--loss-fallthrough", str(sp.get("loss_fallthrough", 1)),
                                 "--gumbel-sigma-mctx", str(sp.get("gumbel_sigma_mctx", 0))])
            else:
                raise
        x, pi, z, shape = read_samples(data)
        full = shape.get("full")
        # Hold-out split BEFORE augmentation: augmenting first makes held-out
        # positions mirror twins of training positions (audit #8). Split raw,
        # then augment the train portion only.
        (train_x, train_pi, train_z, train_full,
         holdout_x, holdout_pi, holdout_z, holdout_full) = split_held_out(
             x, pi, z, full, holdout_frac=0.1)
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
        # Evaluate network on held-out data for monitoring
        net.eval()
        device = next(net.parameters()).device
        with torch.no_grad():
            holdout_x_t = torch.as_tensor(holdout_x, device=device)
            holdout_pi_t = torch.as_tensor(holdout_pi, device=device)
            holdout_z_t = torch.as_tensor(holdout_z, device=device)
            logits_ho, v_ho = net(holdout_x_t)
            # Policy cross-entropy (average over held-out samples, uniform weights)
            per = -(holdout_pi_t * F.log_softmax(logits_ho.float(), -1)).sum(-1)
            heldout_policy_loss = per.mean().item()
            # Value MSE
            heldout_value_mse = F.mse_loss(v_ho.squeeze(), holdout_z_t).item()
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
        effective_steps = min(cfg["training"]["steps_per_iter"], max_allowed_steps)
        # B4 final LR anneal: in the last anneal_frac of wall-clock budget, drop
        # LR by anneal_factor once (both default-off: anneal_frac 0).
        el_pre = time.time() - t0
        tr = cfg["training"]
        if (tr.get("anneal_frac", 0) > 0 and not annealed
                and el_pre > budget * (1.0 - tr["anneal_frac"])):
            set_lr(base_lr * tr.get("anneal_factor", 0.1))
            annealed = True
            print(f"anneal: lr {base_lr} -> {base_lr * tr.get('anneal_factor', 0.1)} at el={el_pre:.0f}s")
        # Train for effective_steps
        losses = {}
        for _ in range(effective_steps):
            try:
                losses = step(*buf.sample(batch_size))
            except TypeError:
                xb, pib, zb = buf.sample(batch_size)[:3]
                losses = step(xb, pib, zb)
        el = time.time() - t0
        log.write(json.dumps(dict(iter=it, elapsed_s=round(el, 1), selfplay=stats,
                                replay=buf.n,
                                heldout_policy=heldout_policy_loss,
                                heldout_value=heldout_value_mse,
                                effective_steps=effective_steps,
                                max_reuse_ratio=max_reuse_ratio,
                                fresh_positions=fresh_positions,
                                evaluator=ev, alpha=round(alpha, 3),
                                **losses)) + "\n")
        log.flush(); it += 1
        save_checkpoint(net, os.path.join(a.out, "latest.pt"))
        try:
            export_torchscript(net, shape, latest_ts)
        except Exception as e:
            print(f"export ts failed: {e}")
        if el >= next_ck:
            ck = os.path.join(a.out, f"checkpoint_{int(next_ck // 60)}m.pt")
            save_checkpoint(net, ck); next_ck += ck_every
            # Cheap sanity arena when possible (rollout evaluators on CPU): new vs random baseline
            try:
                ev_cfg = cfg.get("eval", {})
                arena = sh_json([f"{a.build}/evaluate", "--game", a.game, "--games", str(ev_cfg.get("games", 20)),
                                 "--sims-a", str(ev_cfg.get("sims", 200)), "--sims-b", "50"])
                log.write(json.dumps(dict(iter=it, elapsed_s=round(time.time() - t0, 1), arena_vs_weak=arena, checkpoint=ck)) + "\n"); log.flush()
            except Exception as e:
                print(f"arena failed: {e}")
    save_checkpoint(net, os.path.join(a.out, "latest.pt"))
    # Promote latest to champion (full gating with SPRT is TODO; cheap arena above is the sanity check)
    import shutil
    try:
        shutil.copy(os.path.join(a.out, "latest.pt"), os.path.join(a.out, "champion.pt"))
    except Exception:
        pass

if __name__ == "__main__": main()
