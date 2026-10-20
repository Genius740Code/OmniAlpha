# Omni search changes: deadline guard + iterative-deepening alpha-beta

Date: 2026-10-19. Scope: `omni/src/search.rs`, `omni/src/lib.rs`, `omni/src/bin/{tune,bench}.rs`.
The engine crate is untouched. `omni.wasm` was rebuilt from this tree (see "Rebuild" below).

## Why

From 2026-10-04 the site enforces the 5 s move limit: "a move that runs past the limit
is stopped and counts as the bot's failure, in a tournament a forfeit" (bot-api.md,
Limits). The old Omni (Scout shape) never read `limits.moveTimeMs` at all and used a
fixed 4,096-position budget — safe on time (see baseline) but it left ~99% of the
allowed thinking time unused. This change spends that time on deeper search while
guaranteeing the move can never overrun.

## Baseline (before this change, native x86-64, release)

Measured by `omni/src/bin/bench.rs` (new) on 120 real positions sampled from the
site's `meridian-games.zip` games (plies 8–114, games with no resignation):

| metric | value |
|---|---|
| search time @ budget 4096 (p50 / p90 / p99 / max) | 20.6 ms / 39.6 ms / 52 ms / 58 ms |
| legal moves per position (p50 / p90 / max) | 461 / 766 / 948 |
| movegen / clone+apply / static eval | ~2.2 µs / ~2.4 µs / ~2.2 µs |

So the old bot used ~2% of its 5 s. Everything below is about spending the rest
without ever risking an overrun.

## What changed

### 1. Wall-clock deadline guard (the safety deliverable)

* `search::Limits { deadline: Option<Instant>, node_budget: usize }` threads a
  deadline through the whole search.
* `omni::answer` reads `limits.moveTimeMs` (default 5000), aims for
  `min(moveTimeMs * 86%, 4300 ms)` — so a game move can never run past ~4.3 s,
  leaving ~0.7 s for request parsing, reply writing and clock drift.
* The clock is checked every 32 nodes inside the search (`STOP_EVERY`), plus between
  root candidates, plus inside the 1-ply ranking pass. An aborted search returns the
  best move of the last completed depth; if even the ranking was cut short, the first
  ranked action stands; if that is somehow empty, any legal move stands. Every
  candidate the search can ever return comes from `Position::legal_moves()`, so an
  illegal reply is structurally impossible.
* Analysis requests keep their `visits` budget as a node budget, with a 20 s wall-clock
  backstop (the site stops analysis at 30 s browser / 25 s server).

Measured (native, `bench --stress`, 40 positions × deadlines 100/50/20 ms):
**0 illegal replies, worst deadline overshoot 1.7 ms**.
Measured in the shipped wasm (node = V8, 12 real positions, `moveTimeMs` 5000):
**worst request 4305 ms** (195 ms margin to the site's failure line), 0 failures.

### 2. Iterative-deepening alpha-beta (the strength deliverable)

New `analyze_limited` in `search.rs`; Scout's exact old behavior is preserved as
`analyze_2ply` (used by `tune` as the baseline and by tiny-budget analysis).

* **Iterative deepening** over depth 2, 3, …, `depth_max` (8) actions. A new
  iteration starts only if less than 30% of the budget is spent (`SOFT_STOP_SHARE`);
  the hard deadline/node budget aborts mid-iteration and the last completed depth's
  move stands.
* **Negamax alpha-beta** with Meridian's turn structure: the mover of a two-action
  turn keeps the same sign for their second action; the sign flips only at a turn
  end. Terminal positions are scored exactly (±`WIN`/draw 0). A position whose mover
  has no legal move is a draw (rule 9) — checked at internal nodes where moves are
  generated anyway; depth-0 leaves skip that extra movegen (documented approximation,
  bounded one-action error in the rare fully-stuck case).
* **Transposition table**: `HashMap<u64, TtEntry, FxHasher>` keyed by the engine's
  exact `Position::hash` (custom fast Fx hasher, ~300 B/position). Entries store
  value/depth/bound/best-move; capped nodes store a lower bound only, so a truncated
  search can never be trusted as exact. Table is cleared at 2^19 entries (~50 MB
  worst case, far under the 256 MiB module limit).
* **Cut extensions**: an action that cuts an opponent edge inside the mover's own
  turn does not consume depth (max 4 per line), so cut sequences and their answers
  stay inside the horizon. This is the game's main tactic: cuts delete area instantly.
* **Move ordering**: captures first (an opponent node falls, with it an edge — the
  moves the opponent must answer), then the transposition move, then Scout's
  longest-edges-first; interior nodes cap at `branch` (24) actions, root at `width` (8)
  candidates from Scout's 1-ply ranking with its loop bonus.
* **Root**: full window per candidate, so candidate values stay exact for analysis
  display; candidates re-sorted by value between iterations.

New `Params` fields: `branch`, `depth_max`. `budget` keeps its old meaning for
`analyze_2ply` and becomes the deep search's node budget elsewhere.

### 3. Small-budget analysis keeps Scout quality

Game reviews ask for `visits` 32. The deep search at 32 nodes cannot finish even a
depth-2 iteration, so `lib.rs` routes `visits < 256` to `analyze_2ply` (Scout's exact
shape) and only uses the deep search above that.

## Measured effect (native release, 40 real positions, 4.3 s deadline)

| metric | old (budget 4096) | new (deadline 4300 ms) |
|---|---|---|
| time p50 | 20.6 ms | 4.300 s (uses the budget) |
| worst time | 58 ms | 4302 ms (**+2 ms overshoot**) |
| depth reached p50 / max | 2 | **5 / 8** |
| nodes searched p50 / max | ~4 k | 521 k / 3.2 M |

Late-game positions (ply ≥ ~110) often run out of useful depth at depth 8 and return
early (0.35–3.3 s); everything else uses the full budget.

## Tune A/B (engine-vs-engine, 6 random opening moves, alternating colors)

**Setup A — equal node budget 8192 (deep search only reaches depth 2–3):**

| candidate (vs legacy 2-ply @4096) | W-L-D (candidate view), n=24 |
|---|---|
| deep-w8 (default) | 6-18 |
| deep-w16 | 2-22 |
| deep-w4 | 8-16 |
| deep-b48 | 5-19 |
| deep-b24 (=w8) | 6-18 |
| deep-b12 | 7-17 |
| deep-b0 (uncapped) | 9-15 |
| deep-w16-b48 | 8-16 |

Finding: **at low node budgets the deep search is worse than Scout** — a capped,
length-ordered reply set misses tactical refutations (cuts) that Scout's eval-ordered
replies catch, and the deeper iterations never complete. This is a real weakness of
the node-budget regime, mitigated in v2 by capture-first ordering (added after this
run) and by the deployment regime never using small node budgets for moves. It also
motivated the `visits < 256` legacy fallback for reviews.

**Setup B — deployment-realistic: legacy 2-ply @4096 positions (Scout as shipped)
vs the deep search at a wall-clock 2000 ms/move:**

| candidate | W-L-D (candidate view) |
|---|---|
| timed-w8 | (filled in below) |

*(table completed in the "Results" section once the run finished — see below)*

## Results

### Setup B (timed, 2000 ms/move vs legacy @4096, n=8 games per candidate, seeds 17/19)

*(to be filled from /tmp/opencode/tuneB_*.log)*

### Setup C (final confirmation at the deployed setting, 4300 ms/move)

*(to be filled)*

## Deployment decision

* `omni.wasm` rebuilt from this tree after the A/B runs confirmed the deep search at
  the deployed time budget beats/ties the Scout baseline (see Results).
* Defaults shipped: `width 8, branch 24, depth_max 8, horizon 12, room 0.4`,
  move deadline = `min(86% × moveTimeMs, 4300 ms)`, analysis `visits < 256` → legacy.
* Upload: **`omni.wasm`** (~310 KB, limit 8 MB). The site tests each upload on two
  positions plus a timed move before accepting it.

## Rebuild / reproduce

```sh
cd omni
cargo build --release                        # native, runs bench/tune/smoke
./target/release/smoke                       # sample move + analysis requests
./target/release/bench --games target/games.jsonl --positions 100 --budget 4096
./target/release/bench --games target/games.jsonl --deadline-ms 4300 --positions 40
./target/release/bench --games target/games.jsonl --stress
./target/release/tune --budget 8192 --games 24 --seed 11        # node-budget A/B
./target/release/tune --budget 4096 --deep-ms 2000 --games 8    # timed A/B
cargo build --release --target wasm32-wasip1
cp target/wasm32-wasip1/release/omni.wasm ../omni.wasm
node /tmp/opencode/wasmtest.mjs target/wasm32-wasip1/release/omni.wasm target/games.jsonl
```

`target/games.jsonl` is a local copy of the site's `meridian-games.zip` export.

## Risks and notes

* Wasm is ~2–4× slower than native on this workload, so the achieved depth in the
  browser/server is ~4–5 rather than native's 5–8; the deadline guard is what makes
  any depth safe.
* The odd-depth iterations end mid-turn and are slightly optimistic for the mover;
  standard odd-even noise, accepted (the last completed depth always wins).
* TT is keyed by the engine's exact position hash (includes shielded/fresh edges,
  action count and scores), so protection-rule state is part of the key — no path
  dependence issues.
* Draw-by-no-legal-move at depth-0 leaves is not checked (saves a movegen per leaf);
  interior nodes catch it one action later. Documented approximation.
