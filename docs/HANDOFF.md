# HANDOFF — read this first (for the next coding agent)

Goal (from the original brief): a reusable C++20 game-AI framework — new game = edit `games/<g>/` only — that trains the
strongest possible bot in ~60 min (competition) or indefinitely (long-term). Optimize **Elo per wall-clock second**.

This repo is a **verified foundation**: Phases 1–2 are done and tested; Phase 6 (clock) core is done; everything GPU/NN is
scaffolded but **never executed** (the authoring sandbox had 1 CPU core, no GPU, no CUDA, no PyTorch, no CMake preinstalled).

## Status legend
- DONE+TESTED = built and exercised here. WRITTEN, UNTESTED = code exists, never run. TODO = not started.

| Area | Status | Where |
|---|---|---|
| Game API (C++20 concept `GameLike`) | DONE+TESTED | `engine/core/api.hpp` |
| CMake, game auto-discovery, ASan/UBSan/TSan options | DONE+TESTED (ASan+UBSan run; TSan option present, not run) | `CMakeLists.txt` |
| Template game + Connect-4 (bitboard + reference + undo + mirror symmetry) | DONE+TESTED | `games/` |
| `validate_game` (random playouts, differential vs reference, undo, symmetry, zero-sum, corruption, infinite-game) | DONE+TESTED; a planted diagonal-check bug was caught by the reference diff | `tools/validate_game.cpp` |
| `benchmark_game` (JSON profile) | DONE+TESTED | `tools/benchmark_game.cpp` |
| PUCT MCTS: virtual loss, batched leaves, Dirichlet, temperature, tree reuse, node-pool compaction, FPU | DONE+TESTED (finds wins/blocks, solves take-away) | `engine/mcts/mcts.hpp` |
| Adaptive budget (min/target/max sims, early stop, extend when unstable) | DONE+TESTED (basic) | `engine/mcts/adaptive.hpp` |
| TimeManager (soft/hard, increment, safety margin, emergency mode, never lets min_think cause timeout) | DONE+TESTED (unit) | `engine/time_manager/` |
| Self-play (multi-threaded independent games) + binary sample format v2 | DONE+TESTED (Python reader verified on real output) | `engine/selfplay/`, `tools/selfplay.cpp`, `python/gai/data.py` |
| `evaluate` arena + Elo/CI | DONE+TESTED but only for **uniform/rollout evaluators** (`--model` rejected) | `tools/evaluate.cpp`, `engine/evaluation/elo.hpp` |
| `play` (human/random/mcts, `--tc 60+1` simulated clock) | DONE+TESTED (evaluator = uniform/rollout only) | `tools/play.cpp` |
| PyTorch models (MLP/ResNet), train step (AMP), checkpoint, TorchScript export | WRITTEN, UNTESTED | `python/gai/` |
| Orchestrator (time budget, checkpoint interval) | WRITTEN, UNTESTED; no arena/champion/racing | `python/gai/orchestrator.py` |
| **C++ NN evaluator + GPU batching** | **TODO (critical path)** | see below |
| Transposition table / NN cache | TODO | |
| Symmetry augmentation in training | TODO | `python/gai/data.py` note |
| Arena with NN, champion promotion, checkpoint Elo | TODO | |
| Configuration racing, game profiler -> auto config | TODO | `configs/competition_60m.yaml: race` |
| `./build/train` CLI | TODO (thin wrapper over `python -m gai.orchestrator`) | |
| Long-term mode loop, experiments | TODO | `configs/long_term.yaml` |
| Gumbel/Sequential-Halving, MuZero/EfficientZero | TODO (Phase 8; only after end-to-end works) | |
| PGO, SIMD, lock-free queues | TODO — only if profiling says so | |
| Learned time management / gameplay data logging | TODO (optional) | |

## Speed/strength ideas
See `docs/FAST_TRAINING_IDEAS.md` (early-stop/solver/Gumbel/KataGo/Go-Exploit ideas with sources, tiers and backlog) -- read it before Step 4 below.

## Suggested order for the next agent
1. **Machine setup** (needs a GPU box): `pip install -r python/requirements.txt`; `python -m gai.model` (smoke test, expected `mlp ok / resnet ok`); fix anything that fails — all PyTorch code is untested.
2. **NN evaluator (critical)**. Implement `NNEvaluator<G> : Evaluator<G>` in `engine/inference/`. Interface is final: `evaluate(const State*, EvalResult*, count)`; priors must be softmax over the *legal moves in `G::legal_moves` order*; value from the view of the player to move. Recommended: libtorch loading the TorchScript file from `python/gai/train.py:export_torchscript` (fastest to build), later TensorRT/CUDA graphs only if profiling shows benefit. Gate behind a CMake option (`GAI_LIBTORCH`), keep the CPU-only build working. Hook it into `make_evaluator` in `tools/common.hpp` (`nn:<path>` spec) and enable `--model` in `evaluate`/`play`.
3. **Cross-thread dynamic batching**: today each self-play thread has its own tree and calls the evaluator with batch = `--batch` leaves. For GPU throughput add a shared inference queue: workers enqueue (state, promise), one GPU thread forms batches (max size / max wait µs), pinned memory + a CUDA stream. `MCTS::select_leaf()/finish_leaf()` already support asynchronous use (virtual loss is implemented).
4. **Close the loop**: orchestrator uses `evaluator: nn:models/latest.ts` after the first iteration; add arena vs champion (`./build/evaluate`), promote on score > threshold, write `models/champion.pt`. Log Elo vs wall-clock.
5. Profile (benchmark_game, nvidia-smi, perf), then: NN cache keyed by `G::hash`, transposition table, symmetry augmentation (export symmetry permutations from C++), playout-cap randomization, configuration racing, game profiler -> config.
6. Only then Phase 8 (Gumbel etc.). Benchmark each change as Elo/minute, not raw sims/s.

## Conventions / gotchas
- **Move value == policy index** (`0 <= m < kActionCount`). Games with huge action spaces (e.g. chess 4672) are fine; `EvalResult::priors` is aligned to the *legal move list*, not the action space.
- State must be small & trivially copyable (MCTS copies it every simulation). If a game needs big state, change MCTS to store states in nodes or use undo (the `undo` hook is optional and already validated when present).
- Two players, strictly ids 0/1, zero-sum outcomes (+1/0/-1). Non-alternating turns are supported by MCTS (it compares `mover`) but the validator only reports them. Single-player or >2-player games need engine changes.
- Games are header-only (`game.hpp`, `reference_game.hpp`) for inlining; the brief's `game.cpp/encoding.cpp/symmetry.cpp/tests.cpp` split was intentionally dropped. `config.yaml` per game is currently only documentation (merge TODO).
- Registry: CMake globs `games/*/game.hpp`; namespace must be `game_<dirname>` with structs `Game` (and `RefGame`). Re-run cmake after adding a game dir. The keyword `template` forced the `game_` prefix.
- MCTS value convention: `Node::w` is from the view of `Node::mover` (player who chose the incoming edge). Root value from `root_value()` is the view of the player to move.
- Known simplifications: no transposition table; FPU = parent Q − 0.25; `evaluate` uses fixed random openings (not paired colors); adaptive-search thresholds are hand-set; `apply()` on illegal moves is unchecked.
- Self-play sample file is raw float32 little-endian (format in `engine/selfplay/selfplay.hpp`); fine for competition mode, replace with sharded/compressed storage for long-term runs.
- Build flags: `-march=native` is ON by default (not portable across machines; `-DGAI_NATIVE=OFF`). LTO: `-DGAI_LTO=ON`. Sanitizers: `cmake -B build-asan -DGAI_SANITIZE="address;undefined" -DCMAKE_BUILD_TYPE=Debug`.

## Reference numbers (authoring sandbox: ONE shared Xeon vCPU @2.8GHz, no GPU — NOT representative of your target hardware)
Connect-4, `benchmark_game`: legal_moves 8.6e7/s, apply+hash 6.5e7/s, encode 8.7e6/s, random games 4.2e6/s,
MCTS (uniform eval) ≈4e6 sims/s, MCTS (1 rollout/leaf) ≈1.9e6 sims/s. `selfplay` (rollout eval, 200 sims, 1 thread) ≈ 850 games/s.
Arena smoke test: 800 vs 100 sims (rollout eval) scored 18W-2L over 20 games. Re-measure everything on the real machine; the NN will dominate runtime, not these numbers.
Nothing in this repo has yet demonstrated Elo improvement from training — that is the first thing to prove once step 2–4 are done.

## Quick commands
```
pip install cmake      # if cmake missing
cmake -S . -B build && cmake --build build -j && (cd build && ctest)
./build/validate_game --game connect4        ./build/benchmark_game --game connect4
./build/selfplay --game connect4 --games 100 --sims 200 --out /tmp/d.bin
./build/evaluate --game connect4 --games 50 --sims-a 800 --sims-b 100
./build/play --game connect4 --p0 human --p1 mcts --tc 60+1
cd python && python -m gai.orchestrator --config ../configs/competition_60m.yaml --game connect4 --minutes 2   # untested
```
