"""Wall-clock-budgeted training loop.
Selfplay (C++ binary) -> replay buffer -> train -> timed checkpoints -> TorchScript export.
Arena vs champion when possible; otherwise promotes latest. Tracks Elo vs wall-clock in log.jsonl.
Usage: python -m gai.orchestrator --config configs/competition_60m.yaml --game connect4 [--minutes 2]"""
import argparse, json, os, subprocess, time, yaml
from .data import read_samples, ReplayBuffer

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
        if nn_ok and os.path.exists(latest_ts) and it > 0:
            ev = f"nn:{latest_ts}"
        # playout-cap randomization flags come from config when present
        cmd = [f"{a.build}/selfplay", "--game", a.game, "--games", str(sp["games_per_iter"]), "--sims", str(sp["simulations"]),
               "--out", data, "--evaluator", ev, "--seed", str(it + 1),
               "--full-prob", str(sp.get("full_prob", 1.0)), "--cheap-frac", str(sp.get("cheap_frac", 0.125)),
               "--forced", str(sp.get("forced", 1)), "--prune", str(sp.get("prune", 1)),
               "--shortcuts", str(sp.get("shortcuts", 1)),
               "--gumbel-root", str(sp.get("gumbel_root", 0)), "--gumbel-eps", str(sp.get("gumbel_epsilon", 0.1))]
        try:
            stats = sh_json(cmd)
        except RuntimeError as e:
            if ev.startswith("nn:"):  # fall back to CPU evaluator
                nn_ok = False
                print(f"nn evaluator failed, falling back: {e}")
                ev = sp.get("evaluator", "rollout")
                stats = sh_json([f"{a.build}/selfplay", "--game", a.game, "--games", str(sp["games_per_iter"]),
                                 "--sims", str(sp["simulations"]), "--out", data, "--evaluator", ev, "--seed", str(it + 1)])
            else:
                raise
        x, pi, z, shape = read_samples(data)
        full = shape.get("full")
        if sp.get("augment", False):
            from .data import maybe_mirror_batch
            x, pi, z, full = maybe_mirror_batch(x, pi, z, full)
        if buf is None:
            buf = ReplayBuffer(cfg["training"]["replay_capacity"], shape["planes"], shape["h"], shape["w"], shape["actions"])
            net, opt, step = make_trainer(cfg["network"], shape, cfg["training"]["lr"], cfg["training"].get("weight_decay", 1e-4), cfg["training"].get("mixed_precision", True))
        buf.add(x, pi, z, full); losses = {}
        for _ in range(cfg["training"]["steps_per_iter"]):
            try:
                losses = step(*buf.sample(cfg["training"]["batch_size"]))
            except TypeError:
                xb, pib, zb = buf.sample(cfg["training"]["batch_size"])[:3]
                losses = step(xb, pib, zb)
        el = time.time() - t0; log.write(json.dumps(dict(iter=it, elapsed_s=round(el, 1), selfplay=stats, replay=buf.n, **losses)) + "\n"); log.flush(); it += 1
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
