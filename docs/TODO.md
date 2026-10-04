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

## Killed by our A/B (need new evidence to revive)

- Gumbel prior-noise in `add_root_noise` — lost 1-19 to baseline. Deleted.
- Real Gumbel Top-m + SH (`engine/mcts/gumbel_root.hpp`) as training default —
  flat lineage (10-10), -382 vs rollout-200. Diagnosed: sigma swamps logits ~10:1.
  Needs sigma rescale (verify mctx qtransform) + frozen-net stage-1 test first.

## TODO, ranked

1. [ ] **v2-with-guards rerun** (10min, `results/`) — reproduce monotonic lineage +
    flat held-out at full length. Gate: held-out never rises; else cut ratio < 3.0.
2. [ ] **Position suite ("the ruler")** — solver-labelled C4 positions, top-1
    accuracy vs sims. Unlocks all frozen-net tests with zero training cost.
3. [ ] **Gumbel sigma rescale** — normalize sigma spread to logit spread, then
    stage-1 suite test at 8-128 sims. Kill on no suite win.
4. [ ] **Evaluator-switch validation** — gated (iter 5) vs switch-at-1 vs hybrid
    net+rollout blend on `build-lt`. Prime suspect for "loses to rollout-200".
5. [ ] **Serve-side free Elo** — max-sim serving, checkpoint ensemble, LCB move
    pick, value-swing time management. Zero retraining.
6. [ ] **Value targets** — lambda-returns/soft-Z mixing, WDL head, loss-balance
    tune (value head never improved v1->v3). Suite-gated, 3 seeds.
7. [ ] **Data diversity** — endgame-first curriculum, archive starts, resign +
    calibration, longer tempering, dedup/downweight duplicates.
8. [ ] **Throughput** — lane-batched cross-game GPU inference, GPU-resident
    replay, batch-vs-quality curve. Gate: >=8x positions/s at <10 Elo cost.
9. [ ] **Schedules** — final low-LR anneal, cyclical LR, EMA export, surprise
    weighting, root-temp x c_puct sweep (frozen net).
10. [ ] **Competition plumbing** — profiler->auto-config, ladder with
    alpha-beta/minimax + old checkpoints, opening book, 5-min dry run, dashboard.
11. [ ] **Engineering tests** — encode parity, TorchScript parity, MCTS-vs-minimax
    on template game, golden-seed regression, sample fuzz, throughput budget in CI.
12. [ ] **Deferred** — MuZero/EfficientZero, transformers, PBT, opponent modeling,
    contempt (check rules), Pgx-style GPU-MCTS (rejected by framework design).

## Notes / corrections to external audits

- No bulk "minimal core" deletion: ablate proven features singly (shortcuts are
  provably safe; keep unconditionally).
- B5 net-switch timings are unverified ranges, not a recipe.
- 20-game arenas have CI ~+-150-300: use for kills only on decisive scores
  (e.g. 1-19); otherwise 100+ games or suite metric.
- `/tmp` gets wiped on this box: all run outputs go to `results/` (gitignored).
