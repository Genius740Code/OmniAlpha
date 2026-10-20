# Omni search changes: deadline guard verified, deep search rejected by the A/B

Date: 2026-10-19/20. Scope: `omni/src/search.rs`, `omni/src/lib.rs`, `omni/src/bin/{tune,bench,parity,smoke}.rs`.
The engine crate is untouched. **`omni.wasm` was rebuilt from this tree** (270,773 bytes, sha256
`8ec84dc8cb2a21ecddf3394bc9cc3e64395ab376be8bf8bf86e78a580084d8ad`).

## Outcome in one paragraph

The old Omni (Scout shape) ignored `limits.moveTimeMs` and used a fixed 4,096-position budget —
about 2% of the allowed thinking time, but structurally safe. Two things were built to spend the
rest of the time: a wall-clock deadline guard, and an iterative-deepening alpha-beta search
("deep search") to spend the guard's budget on. The guard works exactly as designed (measured
below). The deep search does not: engine-vs-engine play put it **behind Scout's 2-ply at every
tested setting — 10-32-0 over 42 games, including 0-6 at the deployed 4,300 ms/move**. So the
shipped bot keeps **Scout's exact 2-ply for moves and analysis**, bounded by its position budget
(a reply takes ~20–70 ms in wasm, worst observed 125 ms), with the budget scaled down only if the
site ever asks for a faster reply than its documented 5,000 ms. The deep search and its deadline
guard remain in `search.rs` as library code, exercised by `bench`/`tune`, for future attempts.

## Why

From 2026-10-04 the site enforces the 5 s move limit: "a move that runs past the limit is stopped
and counts as the bot's failure, in a tournament a forfeit" (bot-api.md, Limits). The old shape
never read the limit, so the work's first goal was a guarantee that no reply can overrun; the
second was to spend the freed ~4.9 s on deeper search. The first goal stands; the second was tried
and honestly failed its evaluation.

## Baseline (Scout shape, native x86-64 release, re-measured 2026-10-20)

`bench --games target/games.jsonl --positions 120 --budget 4096`, 120 positions sampled from the
site's games (plies 8–114, no resignations):

| metric | value |
|---|---|
| legacy search time (p50 / p90 / p99 / max) | 20.0 ms / 35.2 ms / 42.5 ms / 44.7 ms |
| legal moves per position (p50 / p90 / max) | 461 / 766 / 948 |
| deep search, same 4096-node budget (p50 / max) | 13.6 ms / 62.6 ms (reaches only depth 2) |

So the shipped shape uses ~1% of the 5 s limit, and its worst-case reply is ~45 ms native,
~125 ms in wasm — 40× inside the limit with no clock involved.

## What was built

### 1. Wall-clock deadline guard (works; ships as library code, not wired into replies)

* `search::Limits { deadline: Option<Instant>, node_budget: usize }` threads a deadline through
  the deep search. The clock is read every 32 nodes, between root candidates, and inside the
  ranking pass; an aborted search returns the best move of the last completed depth, and every
  candidate it can return comes from `Position::legal_moves()`, so an illegal reply is
  structurally impossible.
* Measured (native `bench --stress`, 40 positions × deadlines 100/50/20 ms): **0 illegal replies,
  worst deadline overshoot 0.55 ms**.
* Measured (native `bench --deadline-ms 4300`, 42 positions): **max 4.3013 s — 1.3 ms past the
  4,300 ms deadline**, p50 4.300 s, depth p50 4 / max 7, nodes p50 151 k / max 2.32 M.
* Measured in wasm (deep-wired build, V8, `moveTimeMs` 5000 → aim `min(86%, 4300 ms)`):
  **worst request 4,303 ms**, 697 ms inside the site's failure line, 0 failures.

The guard is the safety net the deep search needed; since the deep search is not used for
replies, the net is currently exercised only by `bench`/`tune`. It is the piece to reuse if the
deep search is ever re-attempted.

### 2. Iterative-deepening alpha-beta (works mechanically; loses games — not shipped)

`analyze_limited` in `search.rs`; Scout's exact old behavior preserved as `analyze_2ply`.

* Iterative deepening over depth 2…`depth_max` (8), negamax alpha-beta with Meridian's turn
  structure (the mover of a two-action turn keeps the sign on their second action), exact
  terminal scores, draw-by-no-legal-move at internal nodes.
* Transposition table keyed by the engine's exact position hash (~300 B/position, capped at 2^19
  entries ≈ 50 MB, far under the 256 MiB module limit); capped nodes store a lower bound only.
* Cut extensions: an action that cuts an opponent edge does not consume depth (max 4/line).
  The engine sets `MoveOutcome::broken` only for cuts of opponent edges (position.rs:511), so the
  extension means the search sees multi-cut sequences and their answers that a fixed 2-ply
  cannot — verified by `parity`: at nominal depth 2 the deep search agrees with Scout on 23/30
  positions and on the other 7 finds lower values for Scout's choices, the signature of a
  refutation Scout misses.
* Move ordering: captures first, transposition move, then longest edges; interior nodes capped
  at `branch` (24), root at `width` (8) candidates from Scout's 1-ply ranking with loop bonus.

### 3. Small-budget analysis

The first cut routed `visits ≥ 256` to the deep search; late-game testing showed that at
256–~2,000 visits the deep search cannot finish its depth-2 iteration (a ply-114 position has
~900 legal moves) and falls back to a **depth-1** reply — worse than Scout at the same cost.
Fix: **all analysis requests use `analyze_2ply`** at the requested `visits`, exactly as the
site's Scout served them.

## The A/B verdict (engine-vs-engine, 6 random opening moves, alternating colors)

Baseline everywhere: legacy 2-ply @4096 positions (Scout as shipped).

**Setup A — equal node budget 8192 (deep reaches only depth 2–3), n=24, seed 11, 2026-10-19:**

| candidate | W-L-D |
|---|---|
| deep-w8 (default) | 6-18 |
| deep-w16 | 2-22 |
| deep-w4 | 8-16 |
| deep-b48 | 5-19 |
| deep-b12 | 7-17 |
| deep-b0 (uncapped) | 9-15 |

**Setup B — wall-clock 2,000 ms/move vs legacy @4096, 2026-10-20 (logs `/tmp/opencode/tuneB_s{17,19}_w8.log`):**

| candidate | seed 17 | seed 19 | combined |
|---|---|---|---|
| timed-w8 (defaults) | 1-5-0 | 3-3-0 | **4-8-0** |

**Setup C — the deployed setting, 4,300 ms/move, 2026-10-20 (logs `/tmp/opencode/tuneC_s{43,47}_w8.log`):**

| candidate | seed 43 | seed 47 | combined |
|---|---|---|---|
| timed-w8 (defaults) | 0-3-0 | 0-3-0 | **0-6-0** |

**Total: 10-32-0 over 42 games.** More time made it worse, not better: at 2,000 ms/move the deep
search reaches depth ~3, at 4,300 ms depth 4–5 with up to 2.3 M nodes — and still lost every
deployed-setting game. This is not noise (a fair coin does 0-6 once in 64; the combined record is
~2.5σ below even).

### Why it loses

Scout's reply pass values **every** legal reply by its immediate eval (all ~460–950 of them,
eval-ordered). The deep search's interior nodes keep only `branch`=24 forcing-ordered replies
(captures first, then longest), so a *quiet* refutation — a short protecting or threatening move
outside the top 24 — is invisible at the depths the deadline buys, while Scout sees it at
depth 2. Cut extensions recover the loud tactics but not the quiet ones; raising the wall-clock
budget deepens the tree without widening those reply sets, which is exactly what Setup C shows.
Fixing it likely means uncapped reply passes (`deep-b0` was Setup A's best at 9-15) or
eval-ordered interior move generation — a different search, not a parameter.

## What shipped

`lib.rs` routes both request types to `analyze_2ply` (Scout's exact search, same defaults:
width 8, horizon 12, room 0.4, budget 4096):

* **Move requests** read `limits.moveTimeMs`: at ≥250 ms (the site sends 5,000) the budget is
  Scout's full 4,096 positions — bit for bit the behavior the site already rates. Below 250 ms
  the budget scales at 4 positions/ms (best effort; a position's cost varies ~10× with game
  stage, so a hard guarantee at absurdly small limits is impossible without a clock, and the
  site's documented limit is 5,000 ms).
* **Analysis requests** use `analyze_2ply` at `min(visits, 4096)`, defaulting to 240 — exactly
  Scout's review behavior (depth 2 at every budget, including the 256–2,000 slice the deep
  path used to degrade to depth 1).

Measured on the shipped `omni.wasm` (V8 via `/tmp/opencode/wasmtest.mjs`):

| check | result |
|---|---|
| 12 real positions, `moveTimeMs` 5000 | 12/12 legal replies, 0 failures |
| worst move request | **63 ms** (4,937 ms inside the limit; worst observed across runs 125 ms) |
| analysis visits 32 / 1000 | depth 2 in 1 ms / 11 ms, 5 candidates each |
| `moveTimeMs` 100 (scaled budget) | worst request 20 ms, 80 ms inside the limit |
| artifact | 270,773 bytes (limit 8 MB), sha256 `8ec84dc8…` |

Upload: **`omni.wasm`**. The site tests each upload on two positions plus a timed move; a
~20–125 ms reply passes with a wide margin. Tick "This bot can analyze positions" as before.

## Rebuild / reproduce

```sh
cd omni
cargo build --release                        # native, runs bench/tune/parity/smoke
./target/release/smoke
./target/release/bench --games target/games.jsonl --positions 120 --budget 4096
./target/release/bench --games target/games.jsonl --deadline-ms 4300 --positions 40
./target/release/bench --games target/games.jsonl --stress
./target/release/parity --games target/games.jsonl --positions 30
./target/release/tune --budget 8192 --games 24 --seed 11          # node-budget A/B
./target/release/tune --budget 4096 --deep-ms 2000 --games 6 --seed 17 --only timed-w8
./target/release/tune --budget 4096 --deep-ms 4300 --games 3 --seed 43 --only timed-w8
cargo build --release --target wasm32-wasip1
cp target/wasm32-wasip1/release/omni.wasm ../omni.wasm
node /tmp/opencode/wasmtest.mjs target/wasm32-wasip1/release/omni.wasm target/games.jsonl
```

`target/games.jsonl` is a local copy of the site's `meridian-games.zip` export;
`/tmp/opencode/wasmtest.mjs` is the harness that drives the module through the site's ABI
(alloc → run → parse) and times real requests.

## Risks and notes

* The shipped reply path is bounded by construction (positions, not a clock): the deadline guard
  is not on the reply path. If the deep search is ever re-enabled, wire it through
  `best_move_limited` and re-run `--stress` first.
* Wasm timing varies with position (late-game positions cost ~10× mid-game ones): budget any
  future time-based work off the measured worst case (~125 ms full budget), not the p50.
* The odd-depth turn-boundary optimism, TT bounds and depth-0 draw approximations documented for
  the deep search are now moot for site replies, but still apply to `tune`/`bench` runs.
* A stronger bot likely needs uncapped or eval-ordered reply sets rather than more time on the
  current shape; the A/B table above is the evidence to start from.
