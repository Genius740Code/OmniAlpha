# OmniAlpha

General game-AI framework (C++20 + PyTorch, AlphaZero-style) where **adding a game = editing `games/<name>/` only**.
Goal: **strongest bot per wall-clock minute** — 60-min competition mode or long-term training.

```
games/<name>/ (rules) -> validate_game -> benchmark_game -> selfplay (MCTS) -> train (python) -> evaluate -> play
```

## Status (Oct 2026, verified on CPU + CPU-libtorch)

- Engine: PUCT MCTS + MCTS-Solver proofs, forced playouts + policy-target pruning, playout-cap
  randomization (`full_search` flag + masked loss), win/block-in-1 shortcuts, can't-be-overtaken
  early stop, eval cache, adaptive search, TimeManager, Elo arena. 4/4 ctest + ASan/UBSan pass.
- Training loop closes end-to-end (selfplay → replay → train → TorchScript → NN-guided selfplay
  → arena promotion). Connect4 10-min pilots: loss 1.95 → 1.57, checkpoints improve monotonically,
  but absolute strength vs rollout baselines is unresolved (20-game CIs). Gumbel-prior-noise
  variant was A/B killed (1-19 vs baseline).
- Needs a GPU box for real strength: `-DGAI_LIBTORCH=ON`, ResNet, 600+ sims, longer runs.

## Layout

`engine/` (core API, mcts, selfplay, inference, evaluation, time_manager) · `games/` (template,
connect4) · `tools/` (validate/selfplay/evaluate/play/benchmark) · `python/gai/` (model, data,
train, orchestrator) · `configs/` (competition_60m, long_term, connect4_10min_*) · `tests/` · `docs/`.

## Quick start

```bash
cmake -S . -B build && cmake --build build -j && (cd build && ctest)
./build/validate_game --game connect4
./build/selfplay --game connect4 --games 100 --sims 200 --out /tmp/d.bin
PYTHONPATH=python python3 -m gai.model   # expect: mlp ok / resnet ok
PYTHONPATH=python python3 -m gai.orchestrator --config configs/connect4_10min_v2.yaml \
  --game connect4 --minutes 10 --out /tmp/c4 --build build-lt
```

NN build (needs libtorch / pip torch CMake config):

```bash
CMAKE_PREFIX_PATH=<torch>/share/cmake cmake -S . -B build-lt -DGAI_LIBTORCH=ON -DGAI_NATIVE=OFF
cmake --build build-lt -j
LD_LIBRARY_PATH=<torch>/lib ./build-lt/evaluate --game connect4 \
  --evaluator-a nn:champion_a.ts --evaluator-b nn:champion_b.ts --games 100 --sims-a 200 --sims-b 200
```

Docs: `docs/HANDOFF.md` (continue here), `docs/FAST_TRAINING_IDEAS.md` (prioritized Elo/min backlog),
`docs/ADDING_A_GAME.md`, `docs/PROJECT_SUMMARY.md`, `docs/REUSE.md` (license notes — check before copying).
