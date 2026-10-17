"""Pick a serve champion among TorchScript checkpoints (issue #2, item 3).

Round-robin arena over candidate ``.ts`` files, with SPRT early-stopping per pair, and a
markdown report (Elo table + winning artifact + exact serve command) written to
``results/CHAMPION.md``. Zero retraining: pure serve-side selection.

Usage:
  PYTHONPATH=python python -m gai.pick_champion --game connect4 --build build-lt \
      --candidates 'results/connect4_60min_v2smart_20261016/*.ts' \
      --out results/CHAMPION.md --games 100 --sims 200
  # ensemble a subset explicitly:
  PYTHONPATH=python python -m gai.pick_champion --game connect4 \
      --spec 'single=nn:a.ts' --spec 'ens=ensemble:nn:a.ts+nn:b.ts+nn:c.ts'
"""
import argparse
import glob
import json
import math
import os
import subprocess


def sh_json(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {p.stderr.strip()[-500:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def rank_candidates(names, pair_results):
    """Aggregate a round-robin into per-candidate standings.

    names: list of candidate names (stable order, defines index -> name).
    pair_results: dict {(i, j): (wins_i, losses_i, draws)} with i < j, from i's view.
    Returns rows sorted by score desc: (name, wins, losses, draws, score),
    score = (wins + 0.5*draws) / games.
    """
    agg = {n: [0, 0, 0] for n in names}
    for (i, j), (w, l, d) in pair_results.items():
        a, b = names[i], names[j]
        agg[a][0] += w; agg[a][1] += l; agg[a][2] += d
        agg[b][0] += l; agg[b][1] += w; agg[b][2] += d
    rows = []
    for n, (w, l, d) in agg.items():
        g = w + l + d
        rows.append((n, w, l, d, (w + 0.5 * d) / g if g else 0.0))
    rows.sort(key=lambda r: r[4], reverse=True)
    return rows


def score_to_elo(score):
    """Elo of a player with expected score `score` vs the field (logistic model)."""
    s = min(max(score, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / s - 1.0)


def _run_pair(build, game, a_spec, b_spec, games, sims, batch, sprt_cfg, seed0):
    """Arena a vs b. With sprt_cfg, batch + early-stop; returns (wins_a, losses_a, draws)."""
    if not sprt_cfg:
        r = sh_json([f"{build}/evaluate", "--game", game, "--games", str(games),
                     "--sims-a", str(sims), "--sims-b", str(sims), "--max-sims", str(sims),
                     "--evaluator-a", a_spec, "--evaluator-b", b_spec, "--seed", str(seed0)])
        return r["wins"], r["losses"], r["draws"]
    from .sprt import sprt
    w = l = d = 0
    seed = seed0
    while w + l + d < games:
        n = min(batch, games - (w + l + d))
        r = sh_json([f"{build}/evaluate", "--game", game, "--games", str(n),
                     "--sims-a", str(sims), "--sims-b", str(sims), "--max-sims", str(sims),
                     "--evaluator-a", a_spec, "--evaluator-b", b_spec, "--seed", str(seed)])
        w += r["wins"]; l += r["losses"]; d += r["draws"]
        seed += 1
        dec, _ = sprt(w, l, d, sprt_cfg["elo0"], sprt_cfg["elo1"], sprt_cfg["alpha"], sprt_cfg["beta"])
        if dec != "continue":
            break
    return w, l, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--build", default="build-lt")
    ap.add_argument("--candidates", default="", help="glob of .ts files (name = file stem)")
    ap.add_argument("--spec", action="append", default=[],
                    help="explicit candidate as name=spec (repeatable, e.g. single=nn:a.ts)")
    ap.add_argument("--out", default="results/CHAMPION.md")
    ap.add_argument("--games", type=int, default=100, help="max games per pair")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--sprt-batch-games", type=int, default=20)
    ap.add_argument("--no-sprt", action="store_true", help="disable SPRT early-stop per pair")
    ap.add_argument("--sprt-elo1", type=float, default=5.0)
    ap.add_argument("--threads", default="2")
    a = ap.parse_args()

    candidates = []
    for s in a.spec:
        name, _, spec = s.partition("=")
        if not spec:
            raise SystemExit(f"--spec must be name=spec, got '{s}'")
        candidates.append((name, spec))
    if a.candidates:
        for path in sorted(glob.glob(a.candidates)):
            candidates.append((os.path.splitext(os.path.basename(path))[0], f"nn:{path}"))
    if len(candidates) < 2:
        raise SystemExit("need at least 2 candidates (via --candidates and/or --spec)")

    names = [n for n, _ in candidates]
    specs = [s for _, s in candidates]
    sprt_cfg = None if a.no_sprt else dict(elo0=0.0, elo1=a.sprt_elo1, alpha=0.05, beta=0.05)
    pair_results, pair_log = {}, []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w, l, d = _run_pair(a.build, a.game, specs[i], specs[j], a.games, a.sims,
                                a.sprt_batch_games, sprt_cfg, seed0=101 + i * 100 + j)
            pair_results[(i, j)] = (w, l, d)
            pair_log.append((names[i], names[j], w, l, d,
                             score_to_elo((w + 0.5 * d) / max(1, w + l + d))))
            print(f"{names[i]} vs {names[j]}: {w}-{l}-{d}")

    rows = rank_candidates(names, pair_results)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("# Serve champion\n\n")
        f.write(f"Game `{a.game}`, {a.games} max games/pair, {a.sims} sims/move, "
                f"SPRT early-stop {'off' if a.no_sprt else 'on'} (elo1={a.sprt_elo1}).\n\n")
        f.write("## Standings\n\n| # | candidate | W-L-D | score | Elo vs field |\n")
        f.write("|---|---|---|---|---|\n")
        for rank, (n, w, l, d, s) in enumerate(rows, 1):
            f.write(f"| {rank} | `{n}` | {w}-{l}-{d} | {s:.3f} | {score_to_elo(s):+.0f} |\n")
        f.write("\n## Head-to-head\n\n| A | B | W-L-D (A) | Elo(A-B) |\n|---|---|---|---|\n")
        for a_n, b_n, w, l, d, elo in pair_log:
            f.write(f"| `{a_n}` | `{b_n}` | {w}-{l}-{d} | {elo:+.0f} |\n")
        champion = rows[0][0]
        champ_spec = dict(candidates)[champion]
        f.write(f"\n## Champion: `{champion}`\n\n```\n{champ_spec}\n```\n\n")
        f.write("Serve command:\n\n```\n"
                f"{a.build}/play --game {a.game} --evaluator '{champ_spec}' --sims {a.sims}\n```\n")
    print(f"wrote {a.out}; champion = {rows[0][0]}")


if __name__ == "__main__":
    main()
