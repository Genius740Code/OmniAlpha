# Search A/B experiments — frozen-net (agent B, round of 2026-10-21)

Goal: raise Elo per minute using ONLY frozen-net search changes (zero training cost).
Method: one knob at a time, ruler = Elo vs `rollout:200` at fixed sims (suite metric,
`results/c4_position_suite.json`), plus nn-vs-nn arenas for pure search-policy A/Bs.
Tooling: NEW build dir `build-b/` (GAI_LIBTORCH=ON) + scratch arena `tools/arena_g.cpp`
(copy of `evaluate.cpp` + `--gumbel-a/b`, `--gumbel-mode-a/b`, `--fpu-a/b`, `--safety-a/b`
knobs; verified byte-identical W/L vs `evaluate` on same args). Raw JSONL:
`results/search_ab_b/*.jsonl`. Net unless stated: `models/champion_c4_60m/checkpoint_60m.ts`.

## 0. Infrastructure findings (affect all future gating)

- **Champion-60m ruler baseline (40g, seed 11):** −61 @8, −26 @32, +61 @128, +70 @320
  vs rollout-200. Crossover ~100 sims. Consistent with suite (+52 @320 for the older
  v7a-1m net). Champion is stronger than the suite's net at every sim count.
- **`evaluate` is NOT deterministic across identical invocations when `--threads 2`**:
  same command twice gave 20-17-3 then 17-22-1 (Δ~70 Elo). Cause: per-thread RNG streams
  (rollout evaluator consumes RNG; game→thread assignment order varies). Present in stock
  `evaluate.cpp` too. Consequence: 40-game numbers are single samples of a noisy
  distribution; only decisive results (way beyond CI) are actionable. For reproducible
  gating use `--threads 1` or 100+ games. All "keep" decisions below respect this.

## 1. Gumbel σ rescale — KILLED (third kill, now with frozen-net evidence)

TODO asked: verify mctx qtransform, rescale σ spread to logit spread, suite-test at
8/32/128. Both σ modes were already implemented in `engine/mcts/gumbel_root.hpp`
(mctx default; spread-matched flag); neither had been Elo-tested. Arena: Gumbel root
(budget = N sims, SH allocation, improved-policy pick) vs plain PUCT same net same N,
`nn-vs-nn`, 40 games, seed 11, both agents identical otherwise:

| Budget | Gumbel mctx (W-L-D) | Elo | Gumbel spread (W-L-D) | Elo |
|---|---|---|---|---|
| 8 | 9-29-2 | **−191** | 9-29-2 | −191 |
| 32 | 8-27-5 | **−180** | 8-29-3 | −203 |
| 128 | 16-21-3 | −44 | 4-32-4 | **−301** |

- Decisive losses at 8 and 32 sims (the budgets where Gumbel is supposed to shine).
- **Why σ rescale cannot fix it:** at low sims all root children have ~equal (unvisited)
  Q → min-max normalized qn ≈ 0 for every child → sigma term ≈ 0 in BOTH modes, so the
  two modes produce near-identical play (9-29-2 twice at 8 sims). The σ spread only
  becomes nonzero once Q differentiates — by which time PUCT already has better move
  quality. The math is verified correct; the failure is structural (exploration via
  Gumbel sampling wastes the tiny budget vs PUCT's focused exploitation at serve time).
- Verdict: Gumbel root stays dead for serving AND for training targets (its improved
  policy loses to visit-count picks head-to-head at 8-32 sims). Do not revive without a
  fundamentally different budget allocation; the mctx-mode stage-1 top-1-accuracy gate
  (80-98%) does not transfer to play strength.

## 2. Root parameter sweep @320 sims vs rollout-200 — NO KEEPER

Baseline (c_puct 1.5, FPU 0.25, noise off): +70 @40g seed 11, +24 @100g seed 13.
All deltas inside CI (±108 @40g, ±68 @100g):

| Knob | 40g seed 11 | 100g seed 13 |
|---|---|---|
| c_puct 1.0 | +52 | — |
| c_puct 2.25 | +107 | **+17** |
| dirichlet-auto (10/nc) | +107 | **−0** |
| c_puct 2.25 + dirauto | — | +7 |
| FPU 0.0 | −26 | — |
| FPU 0.5 | +17 | — |
| dirichlet 0.3 (legacy) | +26 | — |

The two +107s at seed 11 did not survive seed 13 @100 games. c_puct 1.5 stays.
Low-sims regime (32 sims, training-relevant): base −44, c_puct 1.0 −17, c_puct 2.25 −89
— weak trend toward LOWER c_puct at low sims, all inside CI. Worth a 200-game check only
if someone tunes the training sim budget; no action now.

## 3. Serving-side free Elo — single net greedy wins; ensemble only for sims-fixed play

40g, seed 11, vs rollout-200 (reference: single net fixed 320 = +70):

| Config | Elo | Note |
|---|---|---|
| LCB pick β=0.25 @320 | −17 | no |
| LCB pick β=0.5 @320 | +35 | within CI, no |
| 3-net ensemble @320 | **+89** | 2.9× eval cost |
| **single net, tc 30ms equal clock** | **+80** | uses only 4.4ms/move (early stop) |
| ensemble @tc 30ms | +26 | 2.9× cost eats the sims — worse |
| tc 30ms + max-sims 400/600/800 | +52/+26/+52 | capping headroom never helps |

Recommendation for competition mode (wall-clock tc): **single champion net, greedy pick,
no LCB, no ensemble, generous max-sims headroom** (current defaults already optimal).
Checkpoint ensemble (60m + vw2-30m + vw2strict-60m) is a real strength gain at fixed
sims (+89, consistent with prior "ensemble vs single ≥0") — usable in sims-fixed formats
only. Also notable: at equal 30ms/move the net beats rollout-200 by ~+80 while using ⅓
of its clock — the adaptive early stop is a genuine Elo-per-minute win already banked.

## 4. Adaptive-budget tuning — neutral

`SearchLimits.safety_factor` (can't-be-overtaken) swept 0 / 0.6 / 1.0 / 1.5 at tc 30ms
(+61/+44/+79*/+53) and 320 fixed sims (+44/−9*/+44; *arena_g baseline vs evaluate's +70
differ by the run-to-run nondeterminism above). Everything within CI → keep 1.0.
Extend-when-unstable thresholds live hardcoded in `engine/mcts/adaptive.hpp` (agent A's
file) — not testable without an engine edit; left untouched per file boundaries.

## Files touched (agent B only)

- `tools/arena_g.cpp` — NEW scratch arena (gumbel/fpu/safety knobs). Not in CMakeLists
  (built manually to avoid shared-file conflicts).
- `engine/mcts/gumbel_root.hpp` — NOT modified (read-only analysis; both σ modes already
  existed).
- `build-b/` — new build dir; `results/search_ab_b/` — raw JSONL.
- No git commits, per instructions.

## Bottom line

Frozen-net search tuning finds no free Elo beyond what's already banked: the champion's
serve config is already optimal (single net, greedy, early-stop headroom). Gumbel root
is now killed on play strength (not just training lineage). The only positive lever left
on this axis is the previously-known one: more sims at serve time (crossover ~100 sims;
+70 @320). If more serving Elo is wanted, the direction is throughput → higher sim
budgets, not search-parameter changes.
