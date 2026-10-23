# Round fixes — C++ engine scope (agent D)

Scope: `engine/**` (except `engine/mcts/gumbel_root.hpp`, untouched per handoff),
`tools/**`, `tests/**`. All four pre-verified fixes from the round bug-hunt are
implemented, each as a minimal diff, each pinned by new ctest assertions.

Verification: `cmake --build build -j2` clean (no new warnings), `cd build && ctest`
**5/5 passed** (test_mcts extended with 4 new blocks; all pre-existing assertions
unchanged), `./build/validate_game --game connect4` passes, full ASAN+UBSAN run of
`tests/test_mcts.cpp` and of `build/selfplay` (batch-8, lanes-2, gumbel, tactics2,
resign-probe paths) clean. Nothing outside the owned scope was touched; no commits made.

---

## Fix 1 — `advance_root` proven-sign flip (`engine/mcts/mcts.hpp`)

**Bug.** `Node::proven` is stored mover-view for non-root nodes (+1 = win for the
player who moved INTO the node), but the root follows the player-to-move convention
(`backup()`'s root pass, and every `root_proven()` reader: `adaptive.hpp:42-43`,
`gumbel_root.hpp:44`). `advance_root(m)` promoted the child's mover-view proven to
root without flipping, so after any tree-reusing move `root_proven()` reported the
INVERSE of the truth: an actual proven WIN read as −1 and vice versa.

**Effect of the bug (why it mattered).** * `adaptive.hpp:43` early-stops on
proven-loss/draw — for an advanced proven win it took the loss branch (same `break`,
wrong classification, comment semantics inverted). * `gumbel_root.hpp:44` short-circuits
on `root_proven() == 1` — an advanced proven LOSS read as +1 and short-circuited the
Gumbel search into "the winning move" at a lost root. Selfplay never reads
`root_proven()`, so training was unaffected; the damage was confined to arena/serve
paths that reuse trees (play/evaluate/scaling/arena_g all call `advance_root`).

**Fix.** Two lines after `root_ = c` in `advance_root`: flip ±1 proven (draws are
symmetric, kept). Written so it runs *before* the `compact()` check.

**`compact()` audit (spec question).** Not trivially wrong — it is **correct**: it
only ever runs from `advance_root` after the flip, copies the root node verbatim
(already root-convention) and descendants verbatim (mover-view), renumbers only
indices, and `forced_root_` is already −1. Pinned by test: same pile-8 scenario with
`max_nodes = 40` forces compaction inside `advance_root` (134 → 26 nodes) and the
flipped `root_proven() == +1` survives; second advance still reads −1.

**Test** (`tests/test_mcts.cpp`, take-away game — pile 8 is a proven loss for the
mover; replies leave 7/6/5, each a proven win for the opponent):

| step | before fix | after fix |
|---|---|---|
| search pile 8 | `root_proven() == -1` | same (unchanged) |
| `advance_root(take 3)` → pile 5 | `+1` ← wrong (stored mover-view −1) | **`+1`** ← right (flipped) |
| `adaptive_search` there | early-stop via loss branch | early-stop via **proven-win** branch, `r.best == 0` |
| `advance_root(take 1)` → pile 4 | `-1` ← wrong | **`-1`** ← right |

Also asserts `node_count()` unchanged across `advance_root` (subtree kept).

## Fix 2 — shortcut moves reuse the subtree (`engine/selfplay/selfplay.hpp`)

**Bug.** All six shortcut paths (single-move ~129, win/block-in-1 ~143, tactics2 ~160,
and the lane-mode twins ~283/295/310) did `G::apply(s, mv); mcts.set_root(s);` —
`set_root` discards the live tree, so with `reuse_tree` on, every shortcut move threw
away the subtree the previous search had already built below that move.

**Fix.** Same pattern as the searched-move path (line 214/366): keep the
`G::apply(s, mv)` (keeps the loop's `s` in sync; no double apply — `advance_root`
applies to the tree's own `root_state_`), then
`if (cfg.reuse_tree) mcts.advance_root(mv); else mcts.set_root(s);`. Applies to all
six sites. `advance_root` degrades to `set_root` when the child isn't in the tree
(first move, unexpanded root), so the no-tree case is unchanged.

**Tests** (`tests/test_mcts.cpp`). Take-away selfplay, `sims 32`, fixed seed — this
line runs a **mid-game** block-in-1 (pile 5) plus a terminal win-in-1 (pile 3), so
the reuse path actually executes:
* game length / sample count / outcome / shortcut-sample semantics
  (17 plies, 17 samples, P0 win, 2 value-only one-hot samples with q ∈ {0,1}) are
  **unchanged vs the pre-fix baseline** — recorded before the change, asserted after.
* determinism: same seed replays the same game.
* direct regression pin (`sims 16`): the searched position right after the pile-5
  block shortcut now sees the carried subtree, so its root-value sample moves from
  q = −0.625 (fresh tree; still the reuse-off value) to q = **−0.6** — before the
  fix both configs produced −0.625. This is the only externally observable
  footprint of the reuse and pins it.

**E2E sanity measure** (`./build/selfplay --game connect4 --games 200 --sims 64
--reuse-tree 1 --out ...`; the flag is now actually parsed — added `--reuse-tree` to
tools/selfplay.cpp, it was previously accepted-but-ignored). The 200-game wall time
(0.1–0.23 s) is too short to time (±30% swing run-to-run), so load-matched
interleaved 2000-game single-thread runs (identical `-O3` binaries, back-to-back,
5 rounds) are the honest measurement:

| measure | before | after | delta |
|---|---|---|---|
| games/s (interleaved mean of 5) | 937 | 922 | −1.6%, inside the ±7% noise band |
| positions (deterministic, seed 5) | 54 827 | 55 707 | **+1.6%** — games run slightly longer |
| positions/s | 25 697 | 25 697 | **identical** — per-position cost unchanged |
| `--shortcuts 0` control run | 39 625 pos, same timing | **byte-identical games** (fixes 1/3/4a add no overhead; the +1.6% is purely fix-2's changed game paths) |
| sims 256, 500 games | 14 698 pos | 14 378 pos (−2.2%, shorter games) | reuse starts paying at higher sims |
| batch 8 | 55 684 pos | 55 985 pos (+0.5%), timing neutral | |

**Honest verdict vs the expected "modest positive": NOT observed at this workload.**
At sims 64 with the cheap rollout evaluator, carried visits after shortcut moves are
small and the changed game paths cost +1.6% positions, so games/s is flat-to-slightly-
negative; there is no per-position slowdown either (the `--shortcuts 0` control proves
the bookkeeping changes are free). The reuse should pay where evals are expensive
(NN evaluator) and at higher sims (already −2.2% positions at 256), but that is
hypothesis, not measurement — the libtorch build was not benched here.

## Fix 3 — resign-probe scoring (`engine/selfplay/selfplay.hpp` + `tools/selfplay.cpp`)

**Bug.** The `resign_probe` fraction of hopeless positions played on, but nothing
counted later outcomes — the false-resign rate the probe exists to measure never
surfaced.

**Fix.** `GameRecord` gains `resign_probes`, `resign_probe_wins` and a
`probe_resigners` vector (would-be resigner per probe). The resign check was
restructured into a hopeless-outer / coin-inner block that keeps the exact RNG call
order (probe 0 ⇒ identical behavior); probe games record the would-be resigner, and
once `outcome_p0` is final (terminal / resigned / truncated) each probe scores a
FALSE RESIGN iff that side ended up winning (`p == 0 ? outcome_p0 : -outcome_p0 > 0`).
Both the single-game and lanes finalize paths compute it. `tools/selfplay.cpp`
aggregates to the stdout JSON: `"resign_probes"`, `"resign_probe_wins"`,
`"false_resign_rate"`.

**Tests.** Deterministic constants pinned in `tests/test_mcts.cpp` (connect4, rollout
seed 77, rng seed 9, 20 games, `resign_q −0.5`, probe 1.0): probes **14**, false wins
**1** (one "hopeless" side came back to win — proves the scoring fires and the sign
flip is right); lanes path: probes **7**; probe 0: probes 0 and 18/20 games end by
resignation (short games) — the off-switch unchanged. Live tool check:
`--resign-q -0.5 --resign-probe 1.0 --lanes 4` →
`"resign_probes":124,"resign_probe_wins":2,"false_resign_rate":0.0161`.

## Fix 4 — batch dedup + root-expansion note (`engine/mcts/mcts.hpp` + comments)

**(a) Duplicate leaves in one batch — implemented always.** `MCTS::run()` now dedups
the non-terminal leaf states of a batch by `G::hash`, evaluates each distinct position
once and fans the result out to every duplicate leaf (all duplicates still
expand-once and back-up individually — `finish_leaf`'s existing dup guard). Keyed by
hash exactly like the existing `CachingEvaluator` (same collision-risk acceptance);
O(batch²) linear scan, allocation-free after the first batches (vectors hoisted out
of the loop like `leaves`/`sts`/`res`). With a stochastic evaluator this correlates
duplicate leaves' values (identical to what the default CachingEvaluator wrapper
already produced); with a deterministic evaluator nothing changes except work saved.
Semantics are unchanged for `batch == 1` (verified: `--shortcuts 0` control above
produces byte-identical games).

**Test.** The guaranteed-duplicate case: a fresh unexpanded root picked `batch`
times in the first batch is ONE position 8 times. Counting evaluator: 8 leaves
→ **1 evaluated state**, root still gets **8 visits** and is expanded.

**(b) Root expansion as batch-of-1 (`run(1, ev, 1)`) — skipped, per spec "prefer
skip".** Folding it into the next batched `run()` is not trivially safe: it would
drop the +1 visit fresh roots get (sims+1 → sims, shifting every trained-on visit
distribution) and fill the first batch with duplicate root leaves. Left as-is with
a code comment + measurement note at both sites (selfplay.hpp ~170 and ~318): the
expansion is 1 single-state evaluator call per searched position on top of ~sims
leaf evals — ≤1.6% of evals at 64 sims, proportionally less at higher sims; with
reuse_tree the carried visits shrink the batched part, so the fraction rises on
reused roots — revisit if profiling ever shows it hot.

---

## Numbers appendix (all runs this box, `build/`, Release)

* ctest: **5/5** before and after (test_mcts now carries 4 new fix-pinning blocks).
* `validate_game --game connect4`: PASSED (`failures: 0`, 2000 games).
* Baseline sanity (pre-fix, repo binary): 200 games → 5535 positions; after: 5633
  (both 2-thread, unseeded thread interleave, ±noise; the deterministic single-thread
  comparison is the table above).
* Tool smoke post-fix: `evaluate`, `benchmark_game`, `scaling`, `play`, `solve_oracle`,
  gumbel selfplay (`--gumbel 1`), tactics2 selfplay (`--tactics2 1` → 10 solves/20
  games) all work; ASAN+UBSAN clean on test_mcts and on selfplay across
  batch-8 / lanes-2 / gumbel / tactics2 / resign-probe paths.
