# Omni (Meridian bot)

A WebAssembly bot for [Meridian](https://constellation.blueshrimp.uk), forked from
Scout (the site's own bot, `meridian-bot-kit.zip` 2026-10-06). Same engine, same
search shape, tuned parameters. Upload file: **`omni.wasm`** (266KB, limit 8MB).

## What it is

- `engine/` — the site's rules engine, **unmodified** (legal moves, cuts, area, scoring).
- `omni/` — our bot crate (fork of `scout/`):
  - `src/search.rs` — Scout's 2-ply search with a `Params` struct
    (`width`, `horizon`, `room_weight`, `budget`). Defaults = Scout values.
  - `src/lib.rs` — site request/reply + `meridian_abi/alloc/run` exports (unchanged logic).
  - `src/bin/tune.rs` — engine-vs-engine parameter tuner ("train" step).
  - `src/bin/smoke.rs` — feeds sample site requests through `answer()`.
- `omni.wasm` — the built upload artifact.
- `RULES.md` — the rules.

## Training (parameter search, done 2026-10-06)

`tune` plays baseline (Scout defaults) vs each candidate, alternating colors,
candidate point of view. Budgets are reduced for speed; winners re-checked higher.

| round | budget | games/pairing | result (candidate view) |
|---|---|---|---|
| 1 | 128 | 2 | width12 1-1, width6 0-2, horizon8 0-2, horizon16 1-1, room0.2 0-2, room0.6 1-1 |
| 2 | 512 | 4 | width12 **0-4**, width6 2-2, horizon8 2-2, horizon16 2-2, room0.2 2-2, room0.6 2-2 |
| 3 | 1024 | 8 | width12 4-4, width6 4-4, **horizon8 0-8**, horizon16 4-4, room0.2 4-4, **room0.6 0-8** |

Killed: `horizon 8` (0-8) and `room_weight 0.6` (0-8) — both decisive at n=8.
The width12 0-4 at round 2 did not reproduce (4-4 at round 3): noise.
**Conclusion: Scout defaults (width 8, horizon 12, room 0.4) confirmed; shipped
unchanged.** The bot plays at Scout strength under our name/ratings.

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

Limits (per site docs): file 8MB (we're 266KB), memory 256MiB, aim for
`limits.moveTimeMs` (5,000ms in games). Analysis honors its `visits` budget.
