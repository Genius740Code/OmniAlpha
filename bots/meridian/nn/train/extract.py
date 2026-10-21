#!/usr/bin/env python3
"""Incremental, parallel dataset extraction for the Meridian supervised trainer.

Reads the site's games.jsonl, replays every game with the real engine (via the
Rust dumper, which calls `nn::input_planes` itself — planes are byte-exact by
construction), and caches one packed sample file per game, keyed by the sha256
of the game's JSON line. A re-downloaded archive reuses every cached game and
only dumps the new ones, so re-extraction cost is proportional to the delta.

Outputs (under --out, default /tmp/opencode/train2):
  cache/            per-game packed sample files (<sha16>.bin, 442 B per position)
  manifest.json     sha -> {game_id, kind, term, margin, n}
  val_games.json    the validation games for `nnbench --val` (hash-split, 6%)
  corpus.json       paths + counts for train.py
  nn-data/          artifacts at the paths nnbench expects:
                     out3/val_games.json, out/val_dense_head.bin

Usage:
  python3 extract.py [--games bots/meridian/omni/target/games.jsonl]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
DUMPER = os.path.join(REPO, "bots/meridian/nn/train/dumper/target/release/nn-train-dumper")
GAMES_JSONL = os.path.join(REPO, "bots/meridian/omni/target/games.jsonl")

REC = 442  # bytes per packed sample (see dumper/src/main.rs)
VAL_PCT = 6  # hash-split fraction of games held out for validation
VAL_CURRICULUM = 6  # opening_len written into val_games.json (skip first 6 plies)
DENSE_MAX = 2048  # samples for the nnbench --check parity artifacts


def sha16(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=GAMES_JSONL)
    ap.add_argument("--out", default="/tmp/opencode/train2")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rebuild", action="store_true", help="ignore and refresh the cache")
    args = ap.parse_args()

    out = args.out
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")
    manifest = {} if args.rebuild or not os.path.exists(manifest_path) else json.load(open(manifest_path))

    # 1. Hash every line; keep only games not already cached.
    t0 = time.time()
    games = []  # (sha, parsed json, raw line)
    with open(args.games) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            g = json.loads(line)
            g["_sha"] = sha16(line)
            games.append((g["_sha"], g, line))
    corpus_sha = hashlib.sha256(("".join(s for s, _, _ in games)).encode()).hexdigest()[:16]

    # Drop stale manifest entries whose files vanished.
    manifest = {k: v for k, v in manifest.items() if os.path.exists(os.path.join(cache, k + ".bin"))}

    fresh = [(s, g, l) for (s, g, l) in games if s not in manifest]
    print(f"corpus: {len(games)} games (sha {corpus_sha}), {len(games) - len(fresh)} cached, {len(fresh)} to dump")

    # 2. Dump the fresh games with N parallel dumper processes.
    if fresh:
        shards = [[] for _ in range(max(1, args.workers))]
        for i, item in enumerate(fresh):
            shards[i % len(shards)].append(item)
        procs = []
        for w, shard in enumerate(shards):
            if not shard:
                continue
            tmp = os.path.join(out, f".fresh.{w}.jsonl")
            with open(tmp, "w") as fh:
                for _, g, _ in shard:
                    fh.write(json.dumps(g, separators=(",", ":")) + "\n")
            log = os.path.join(out, f".fresh.{w}.log")
            procs.append((w, shard, subprocess.Popen(
                [DUMPER, "--games", tmp, "--out", cache],
                stdout=open(log, "w"), stderr=subprocess.STDOUT)))
        for w, shard, p in procs:
            p.wait()
            if p.returncode != 0:
                sys.exit(f"dumper shard failed with code {p.returncode}")
            # Re-read the dumper's per-game metadata and fold it into the manifest.
            for (s, g, _), meta in zip(shard, open(os.path.join(out, f".fresh.{w}.log"))):
                meta = json.loads(meta)
                if not meta.get("ok") and meta.get("err") == "parse":
                    print(f"  skipping unparsable game {g.get('id')}", file=sys.stderr)
                    continue
                manifest[s] = {
                    "game_id": g.get("id"),
                    "kind": g.get("kind"),
                    "term": g.get("termination"),
                    "margin": g["score"]["blue"] - g["score"]["red"],
                    "n": meta["n"],
                    "ok": meta["ok"],
                }
        for w, _, _ in procs:
            os.unlink(os.path.join(out, f".fresh.{w}.jsonl"))
        json.dump(manifest, open(manifest_path, "w"))
    dt = time.time() - t0
    total_n = sum(v["n"] for v in manifest.values())
    print(f"extraction: {len(manifest)} games cached, {total_n} positions, {dt:.1f}s "
          f"({len(fresh)} games dumped this run)")

    # 3. Deterministic validation split by content hash, in corpus order.
    val = [(s, g, l) for (s, g, l) in games if int(s, 16) % 100 < VAL_PCT]
    val_games_json = os.path.join(out, "val_games.json")
    with open(val_games_json, "w") as fh:
        for _, g, _ in val:
            fh.write(json.dumps({
                "moves": g["moves"],
                "opening_len": VAL_CURRICULUM,
                "margin": g["score"]["blue"] - g["score"]["red"],
                "game_id": g.get("id"),
            }) + "\n")
    # nnbench expects one json array (and looks for it at /tmp/opencode/nn-data/out3).
    arr = [json.loads(l) for l in open(val_games_json)]
    nn_data = os.path.join(out, "nn-data")
    os.makedirs(os.path.join(nn_data, "out"), exist_ok=True)
    os.makedirs(os.path.join(nn_data, "out3"), exist_ok=True)
    json.dump(arr, open(os.path.join(nn_data, "out3", "val_games.json"), "w"))
    shutil.copy(val_games_json, os.path.join(nn_data, "val_games.jsonl"))

    # 4. Dense parity artifacts for `nnbench --check`: the first DENSE_MAX
    #    validation decision positions, in val_games.json order.
    dense_games = os.path.join(out, ".dense_games.jsonl")
    with open(dense_games, "w") as fh:
        for _, g, _ in val:
            fh.write(json.dumps(g, separators=(",", ":")) + "\n")
    dense_out = os.path.join(nn_data, "out", "val_dense_head.bin")
    subprocess.run([DUMPER, "--dense-games", dense_games, "--dense-out", dense_out,
                    "--dense-max", str(DENSE_MAX), "--skip", str(VAL_CURRICULUM)], check=True)
    os.unlink(dense_games)

    json.dump({
        "corpus_sha": corpus_sha,
        "games": len(games),
        "games_cached": len(manifest),
        "positions": total_n,
        "val_games": len(val),
        "val_positions": sum(manifest.get(s, {}).get("n", 0) for s, _, _ in val),
        "cache": cache,
        "manifest": manifest_path,
        "val_games_json": val_games_json,
        "nn_data": nn_data,
        "margin_scale_note": "targets stored as raw margins; tanh applied at train time",
    }, open(os.path.join(out, "corpus.json"), "w"), indent=1)

    margins = sorted(v["margin"] for v in manifest.values())
    import statistics
    print(f"val: {len(val)} games; margins p50 {statistics.median(margins):+.0f} "
          f"close(|m|<=300) {sum(1 for m in margins if abs(m) <= 300)} games")


if __name__ == "__main__":
    main()
