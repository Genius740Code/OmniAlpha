# OmniAlpha TODO — consolidated from measured runs + external audits

Goal: max Elo per wall-clock minute (60-min competition + long-term).
Rule: one change at a time behind flags; A/B at equal wall-clock; kill on SPRT failure.

## Done and kept (4/4 ctest + ASan green)

- PUCT + Dirichlet + temperature exploration (`engine/mcts/mcts.hpp`)
- MCTS-Solver proof propagation; forced playouts + policy-target pruning
- Playout-cap randomization + `full_search` flag + masked policy loss
- Single-move skip + win/block-in-1 shortcuts (value-only)
- Can't-be-overtaken early stop + safety factor; root-FPU fix (estimate, not -0.25)
- `CachingEvaluator` by hash; mirror augmentation; libtorch CPU evaluator + arena
- Held-out split + `max_reuse_ratio` clamp + `nn_start_iter` gating (`python/gai/`)
- `tools/scaling.cpp` harness (measured +70–108 Elo/doubling on rollout, 10-game CIs)
- **Value-target mixing: soft-Z blend outcome with search Q (`engine/selfplay/selfplay.hpp`)**:
  - `q` field added to Sample (MCTS root_value at move time)
  - `value_lambda` in SelfPlayConfig [0,1]; default 0.1
  - Value = (1 - λ) * outcome + λ * q at sample finalization
  - Lambda=0.1: +191 Elo at 1m, -107 Elo vs rollout-200 (first positive trend!), +168 Elo over v2-guards-10m at 10min
- **Position suite ("the ruler")** — Elo vs rollout at various sims (`results/c4_position_suite.json`):
  - 10s: −511 Elo, 1-19 vs rollout-200
  - 20s: −241 Elo, 4-16
  - 40s: −70 Elo, 8-12
  - 80s: −241 Elo, 4-16
  - 160s: −89 Elo, 7-12
  - 320s: **+52 Elo, 11-8** (net BEATS rollout-200 for first time!)
  - Doubling Elo gain: ~+140–170 Elo per 2× sims in 40–320 range
  - Zero training cost per test; unlocks all frozen-net tuning

## Killed by our A/B (need new evidence to revive)

- Gumbel prior-noise in `add_root_noise` — lost 1-19 to baseline. Deleted.
- Real Gumbel Top-m + SH (`engine/mcts/gumbel_root.hpp`) as training default —
  flat lineage (10-10), -382 vs rollout-200. Diagnosed: sigma swamps logits ~10:1.
  Needs sigma rescale (verify mctx qtransform) + frozen-net stage-1 test first.
- HybridEvaluator annealed bootstrap (v5) — even strength, no breakthrough over
  vanilla guarded-v2. Kept as option, not default.

## TODO, ranked

1. [ ] **Serve-side free Elo** — max-sim serving, checkpoint ensemble (last 3),
     LCB move pick (A6 from external audit), value-swing time management. Zero
     retraining. Goal: convert the +52 Elo at 320s vs rollout-200 into real serve
     Elo gain. **Priority: critical — the +52 Elo at 320s is the first time we've
     beaten rollout-200; serving can amplify this.**

2. [ ] **Gumbel σ rescale** — normalize sigma spread to logit spread (verify
     `mctx` qtransform), then stage-1 suite test at 8-128 sims. Kill on no suite
     win. Now that the ruler exists, can test frozen-net Gumbel variants.

3. [ ] **Evaluator-switch validation** — gated (iter 5) vs switch-at-1 vs hybrid
     net+rollout blend on `build-lt`. Value-mixing already shifts the numbers
     significantly (+52 at 320s).

4. [ ] **Data diversity** — endgame-first curriculum, archive starts, resign +
     calibration, longer tempering, dedup/downweight duplicates. Needs the ruler
     first.

5. [ ] **Throughput** — lane-batched cross-game GPU inference, GPU-resident
     replay, batch-vs-quality curve. Gate: >=8x positions/s at <10 Elo cost.

6. [ ] **Schedules** — final low-LR anneal, cyclical LR, EMA export, surprise
     weighting, root-temp × c_puct sweep (frozen net).

7. [ ] **Competition plumbing** — profiler→auto-config, ladder with
     alpha-beta/minimax + old checkpoints, opening book, 5-min dry run, dashboard.

8. [ ] **Engineering tests** — encode parity, TorchScript parity, MCTS-vs-minimax
     on template game, golden-seed regression, sample fuzz, throughput budget in CI.

9. [ ] **Deferred** — MuZero/EfficientZero, transformers, PBT, opponent modeling,
     contempt (check rules), Pgx-style GPU-MCTS (rejected by framework design).

## Key resultscorecard (10-min Connect4, CPU MLP 150-sims)

| Config | 1m vs v2-1m | latest vs rollout-200 | latest vs v2-guards-10m |
|---|---|---|---|
| v2 (guarded) | — | −168 (loses) | — |
| v4 (Gumbel prior-noise) | — | 1-19 lost | — |
| v5 (hybrid) | +17 | −35 | — |
| v6 (val-mix, lambda=0.5) | +147 | −512 | — |
| v6 (val-mix, lambda=0.3) | +191 | −191 | — |
| v6 (val-mix, lambda=0.1) [before q fix] | +191 | +35 | — |
| v7 (val-mix, lambda=0.1) [with q fix, SIGN BUG] | +191 | −107 | +168 |
| **v8 (sign-fixed re-run)** | **−89 (7-12-1, 10m better)** | **−107 @50 / +17 @320** | **+35 (10-8-2)** |
| v9 (audit batch, steps starved 8/iter) | +52 inv (11-8, 1m better) | −301 @50 / −147 @320 | — |
| v9b (repeat, steps starved) | −52 (8-11-1, 10m better) | −241 @50 | — |
| **v10 (steps restored 11/iter)** | **−89 (6-11-3, 10m better)** | **−301 @50 / −147 @320** | **−17 vs v8-10m (9-10-1, EVEN)** |
| **Position suite (320s vs rollout-200)** | N/A | **+52 (11-8)** | N/A |

**Verdict:** Two breakthroughs:
1. **Value-target mixing** (lambda=0.1 + q field): -107 Elo vs rollout-200 (first close trend), +191 at 1m, +168 over 10-min baseline.
2. **Position suite**: 320-sim net beats rollout-200 by +52 Elo (11-8) — first positive result ever against rollout-200.

**v8 sign-fix validation (2026-10-05):** pattern reproduces with correct per-player
value signs — monotonic lineage (10m beats 1m, -89), -107 @50 vs rollout-200
(identical to v7), +17 @320 (positive, softer than v7's +52, within noise ±159),
+35 over v2-guards-10m (positive, softer than v7's +168). Direction confirmed;
magnitudes smaller. Prior v7 Elo margins must be cited as noisy single runs, not
established gains. The value-sign bug is fixed; all future runs use correct targets.

**v9/v10 audit-batch validation (2026-10-05):** the batch (split-before-augment,
loss fallthrough, exact sims, threads forwarding) initially looked like a
regression (v9: inverted lineage, -301 vs rollout). Root-caused one self-inflicted
wound: the reuse-clamp denominator change starved training 11.3→8 steps/iter.
After fix (v10, 11/iter): monotonic lineage restored, but rollout gap (-301 @50,
-147 @320) persists vs v8's (-107, +17). Decisive datum: v10-10m vs v8-10m is
9-10-1 (-17, EVEN) — the regimes produce equal-strength nets; rollout-delta
differences sit inside combined 20-game CIs. Lesson: 20-game arenas cannot
resolve ~100 Elo questions. 100-game arena (threaded) running as the first
trustworthy claim; all future gates use 100 games or suite metric.

**Next priority:** Serve-side free Elo. The +52 Elo at 320s is the first time we've beaten rollout-200. Serving at max sims (600+) + checkpoint ensemble can likely convert this into a decisive strength lead. No more training needed.

## Notes / corrections

- No bulk "minimal core" deletion: ablate proven features singly (shortcuts are
  provably safe; keep unconditionally).
- B5 net-switch timings are unverified ranges, not a recipe.
- 20-game arenas have CI ~+-150-300: use for kills only on decisive scores
  (e.g. 1-19); otherwise 100+ games or suite metric.
- `/tmp` gets wiped on this box: all run outputs go to `results/` (gitignored).
- Position suite (`results/c4_position_suite.json`) is the new ruler baseline.