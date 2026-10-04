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

## Killed by our A/B (need new evidence to revive)

- Gumbel prior-noise in `add_root_noise` — lost 1-19 to baseline. Deleted.
- Real Gumbel Top-m + SH (`engine/mcts/gumbel_root.hpp`) as training default —
  flat lineage (10-10), -382 vs rollout-200. Diagnosed: sigma swamps logits ~10:1.
  Needs sigma rescale (verify mctx qtransform) + frozen-net stage-1 test first.
- HybridEvaluator annealed bootstrap (v5) — even strength, no breakthrough over
  vanilla guarded-v2. Kept as option, not default.

## TODO, ranked

1. [ ] **Position suite ("the ruler")** — solver-labelled C4 positions, top-1
    accuracy vs sims. Unlocks all frozen-net tuning (c_puct × root-temp sweep,
    Gumbel σ rescale validation, LCB serving). Zero training cost per test after
    built. **Priority: critical — needed before any more frozen-net experiments.**

2. [ ] **Serve-side free Elo** — max-sim serving, checkpoint ensemble (last 3),
    LCB move pick (A6 from external audit), value-swing time management. Zero
    retraining. Goal: convert the +35 Elo trend vs rollout-200 into real serve
    Elo gain.

3. [ ] **Gumbel σ rescale** — normalize sigma spread to logit spread (verify
    `mctx` qtransform), then stage-1 suite test at 8-128 sims. Kill on no suite
    win. Parked until ruler exists.

4. [ ] **Evaluator-switch validation** — gated (iter 5) vs switch-at-1 vs hybrid
    net+rollout blend on `build-lt`. Was prime suspect for "loses to rollout-200"
    but value-mixing already shifts the numbers significantly.

5. [ ] **Data diversity** — endgame-first curriculum, archive starts, resign +
    calibration, longer tempering, dedup/downweight duplicates. Needs the ruler
    first.

6. [ ] **Throughput** — lane-batched cross-game GPU inference, GPU-resident
    replay, batch-vs-quality curve. Gate: >=8x positions/s at <10 Elo cost.

7. [ ] **Schedules** — final low-LR anneal, cyclical LR, EMA export, surprise
    weighting, root-temp × c_puct sweep (frozen net).

8. [ ] **Competition plumbing** — profiler→auto-config, ladder with
    alpha-beta/minimax + old checkpoints, opening book, 5-min dry run, dashboard.

9. [ ] **Engineering tests** — encode parity, TorchScript parity, MCTS-vs-minimax
    on template game, golden-seed regression, sample fuzz, throughput budget in CI.

10. [ ] **Deferred** — MuZero/EfficientZero, transformers, PBT, opponent modeling,
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
| v7 (val-mix, lambda=0.1) [with q fix] | +191 | −107 | +168 |

**Verdict:** value-target mixing with lambda=0.1 + proper q values is the first
change that moves the needle toward beating rollout-200. The -107 vs rollout-200
is the best yet (first negative-but-close rather than large negative), and +168
over the 10-min guarded baseline is a real strength gain.

## Notes / corrections

- No bulk "minimal core" deletion: ablate proven features singly (shortcuts are
  provably safe; keep unconditionally).
- B5 net-switch timings are unverified ranges, not a recipe.
- 20-game arenas have CI ~+-150-300: use for kills only on decisive scores
  (e.g. 1-19); otherwise 100+ games or suite metric.
- `/tmp` gets wiped on this box: all run outputs go to `results/` (gitignored).