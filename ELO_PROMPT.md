# Mission

Clone https://YOUR_REPO_URL_OMNIZERO and find every way to make it gain Elo **faster per wall-clock minute**. Stronger bot in less time. How to make training way faster, and ideas the author maybe missed.

# Repo context (pre-load so you don't waste time re-discovering)

OmniAlpha: general AlphaZero-style game AI framework. C++20 engine + libtorch/PyTorch training.
Pipeline: `games/<name>/` rules → MCTS selfplay → replay → train → TorchScript export → NN-guided selfplay → Elo arena promotion.
Blind goal: **strongest bot per wall-clock minute** (60-min competition mode + long-term mode).
Current state: works end-to-end on CPU; Connect4 pilots show monotone checkpoint improvement but unresolved strength vs rollout baselines (20-game CIs too weak). Gumbel-prior-noise variant was A/B tested and killed. Needs GPU box for real strength: `-DGAI_LIBTORCH=ON`, ResNet, 600+ sims.
Everything must keep 4/4 ctest + ASan/UBSan passing.

# Step 0 — read these BEFORE proposing anything

- `docs/HANDOFF.md` — where the project is now
- `docs/FAST_TRAINING_IDEAS.md` — existing prioritized Elo/min backlog
- `docs/TODO.md`, `docs/PROJECT_SUMMARY.md`
- All yaml in `configs/`

**Do not propose items already in FAST_TRAINING_IDEAS.md / TODO.md.** Attach those with your new ideas ranked ABOVE or BELOW existing ones, with reasons why yours outrank them.

# Step 1 — profile first, guess never

1. Build (`cmake -S . -B build && cmake --build build -j`).
2. Profile actual bottlenecks: run selfplay with perf/valgrind/hot-branch analysis of the MCTS loop; profile the python train step; measure % wall-time in (selfplay / train / eval / export / orchestration glue).
3. Quantify: sims/s in selfplay, samples/s in training, time from data → promoted checkpoint end to end.
Before/after numbers required for every optimization claim. No "should help" without a measurement.

# Step 2 — hunt categories (go wide AND deep)

## Search speed (MCTS)
- NN inference batching: is selfplay batching requests? parallel games × batched evals?
- CPU caches, movegen hot path, redundant work between nodes, Zobrist keys, unmake vs copy, node size reduction, memory layout (SoA), lockfree/worker wakeup, parallel root.
- Sim count: are 200 sims wasteful early when net is policy good? adaptive sims already noted — push it harder per timestep.
- Win/block-in-1 shortcut pruning: extendable to mate-in-N, can't-be-overtaken generalization.

## Training throughput
- DataLoader overlap, prefetch, pin memory, device transfer hiding.
- Batch size, LR schedule (warmup + cosine), optimizer choice (AdamW / NAdamW / Lion, β2≈0.98) — pick by measured Elo, not loss.
- Residual net size sweep (channels/depth) vs Elo/min; fp16/quantization; distillation from UCT-only games; net pruning.
- Replay buffer: sample weighting, recency window, dedup; verify generate time ≈ train time (balanced pipeline, no idle phase).
- Early stopping by Elo, arena run concurrently with training (no idle GPU/CPU).
- Checkpoint/export cadence — champ+challenger promotion overhead eating minutes?

## Config
- Config sweep across `configs/connect4_10min_*.yaml` variants; note what config best Elo/min
- Quick run: set sims 128→64, does Elo/min improve when selfplay batch-time kept?

# Step 3 — output format required

Markdown report with two tables, then implementation:

**Table A — new ideas (not in existing backlog):** idea | est. Elo impact (confidence) | effort (S/M/L) | proof needed to validate.

**Table B — existing backlog (FAST_TRAINING_IDEAS.md) re-ranked given your profiling**, with anything you'd now DEprioritize.

**Implementation phase:** take the top 3 cheapest-highest-impact ideas, implement surgically (minimal diffs, hypothesize-then-measure), re-run benchmark, post before/after table. ctest must still pass 4/4.

Report final section: "top 3 missed big-level algorithm ideas" — anything from modern AZ / Leela / KataGo repertoire (batched pooled PV, symmetry augmentation, staged/early-exit nets, sparse simulation policy weighting, robust Gumbel AlphaZero variants that fix the variance problem that killed the earlier Gumbel attempt) — with reasoning AND a small validation experiment each.
