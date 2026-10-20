# Omni (Meridian bot)

A WebAssembly bot for [Meridian](https://constellation.blueshrimp.uk), forked from
Scout (the site's own bot, `meridian-bot-kit.zip` 2026-10-06). Same engine, same
search shape, tuned parameters. Upload file: **`omni.wasm`** (271KB, limit 8MB).

## What it is

- `engine/` — the site's rules engine, **unmodified** (legal moves, cuts, area, scoring).
- `omni/` — our bot crate (fork of `scout/`):
  - `src/search.rs` — Scout's 2-ply search with a `Params` struct
    (`width`, `horizon`, `room_weight`, `budget`). Defaults = Scout values.
    Also holds a deadline-guarded iterative-deepening deep search that lost
    the engine-vs-engine A/B (10-32-0 vs Scout's 2-ply) and is therefore
    **not used for replies** — see `docs/meridian/OMNI_SEARCH_CHANGES.md`.
  - `src/lib.rs` — site request/reply + `meridian_abi/alloc/run` exports.
    Moves and analysis run Scout's exact 2-ply; the move budget scales down
    only if `limits.moveTimeMs` is below 250 ms.
  - `src/bin/tune.rs` — engine-vs-engine parameter tuner ("train" step).
  - `src/bin/bench.rs` — timing/depth benchmark on real site positions
    (`--budget`, `--deadline-ms`, `--stress`).
  - `src/bin/parity.rs` — deep-search vs 2-ply move agreement check.
  - `src/bin/smoke.rs` — feeds sample site requests through `answer()`.
- `omni.wasm` — the built upload artifact.
- `RULES.md` — the rules.

## Training (parameter search, 1-hour run 2026-10-06)

`tune` plays baseline (Scout defaults) vs each candidate, alternating colors,
candidate point of view. The pre-1h tuner was deterministic (same 2 lines
replayed), so it was fixed first: every game now starts from 6 random opening
moves from a seeded RNG (`--openings/--seed`), giving independent games.

| round | budget | games/pairing | seed | result (candidate view) |
|---|---|---|---|---|
| 1 (screen) | 2048 | 80 | 1 | width12 43-37, width16 43-37, horizon10 43-37, rest 38-42…42-38 — except **room0.0 24-56** |
| 2 (verify) | 4096 (=`MOVE_BUDGET`) | 40 | 2 | width12 23-17, rest ≈20-20 — except width6 13-27, width4 12-28, room0.6 15-25, **room0.0 10-30** |

Log: `/tmp/opencode/meridian-1h.log` (~42 min compute on 2 cores).

Killed: `room_weight 0.0` (combined 34-86 over 120 — no room term loses
decisively at both budgets). Narrow widths hurt at full budget (width6/width4
a combined 25-55 in round 2). Nothing beats Scout defaults significantly:
best is width12 at 66-54 combined (+6 over 60-60, ~1σ, n=120) — noise.
**Conclusion unchanged: Scout defaults (width 8, horizon 12, room 0.4)
confirmed; `omni.wasm` ships with them.** Earlier rounds (deterministic
openings, kept for history): horizon8 0-8 and room0.6 0-8 at n=8 did not
reproduce with diverse openings.

Why not a neural net: Meridian's policy space is 17,328 move IDs with
thousands of legal moves per position — AlphaZero-style training on CPU for an
hour cannot beat this search. Tuning the proven search is the Elo/min winner.

## Reproduce

```sh
cd omni
cargo build --release --bin tune
./target/release/tune --budget 1024 --games 8   # ~30s
cargo run --release --bin smoke                  # sample move + analysis requests
cargo build --release --target wasm32-wasip1     # -> target/wasm32-wasip1/release/omni.wasm
cp target/wasm32-wasip1/release/omni.wasm ../omni.wasm
```

Requires Rust 1.85+ and `rustup target add wasm32-wasip1`.

## Upload

1. Open **My bots** → **Add a bot** → **WebAssembly**.
2. Upload `omni.wasm`. The browser tests it on two positions + a timed move
   and shows anything it printed.
3. Each upload becomes a version (SHA-256 named); you can switch back to older ones.
4. Tick **This bot can analyze positions** — it answers `analysis` (up to 5
   candidates, points, depth 2), so it can also review finished games.

Limits (per site docs): file 8MB (we're 271KB), memory 256MiB, aim for
`limits.moveTimeMs` (5,000ms in games) — moves reply in ~20–70ms (worst
observed 125ms), and the budget scales down for any limit below 250ms.
Analysis honors its `visits` budget.
