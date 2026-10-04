# gai — Project Summary & Handoff

A complete record of what was built, what was verified, what is still missing, which ideas the design draws on,
and how the next agent should continue. Companion to `docs/HANDOFF.md` inside the repo (that file is the short
operational version; this one is the full story).

---

## 1. The original goal

Build a reusable C++ framework where **adding a new game means editing only `games/<name>/`**, then train the
strongest possible bot for it:

- **Competition mode:** best playing strength after ~60 minutes of training.
- **Long-term mode:** keep training/improving for hours or indefinitely.
- Guiding rule: *optimize Elo per wall-clock second*, not elegance or raw throughput.
- Stack requested: C++20/23, CMake, CUDA, Python, PyTorch; hot loops in C++, orchestration in Python.

## 2. Environment reality (important context)

The authoring sandbox had: **1 CPU core, 4 GB RAM, no GPU, no CUDA, no PyTorch, no CMake (installed via pip), g++ 13.3.**
Consequences:
- Everything CPU-side was built and tested for real.
- Everything GPU/NN-side was written (Python) or only designed (C++) and **never executed**.
- All performance numbers are from a single shared vCPU and are not representative of target hardware.
- Network access was restricted: the GitHub API returned nothing, so open-source projects could **not** be inspected.

## 3. What we did (chronological)

1. Inspected the environment and reported the limits above; agreed to build a foundation for another agent to finish.
2. Designed a compile-time game API (C++20 concept) so games are zero-overhead templates, not virtual classes.
3. Built the MCTS engine, evaluator interface, adaptive search, time manager, self-play, and Elo estimation.
4. Wrote two games: a tiny **template** game (Take-away) and **Connect-4** (bitboard + naive reference).
5. Built five CLI tools and a CMake system that auto-discovers games.
6. Wrote tests, ran them, ran a **mutation test** (planted a bug to prove the validator catches it), ran ASan+UBSan.
7. Wrote Python scaffolding (models, training step, orchestrator, data reader) and the two YAML configs.
8. Verified the Python data reader against real C++ self-play output.
9. Wrote docs, committed to git, zipped, and re-built from the zip in a clean directory (all tests pass).

## 4. Repository layout

```
gai/
├── CMakeLists.txt              build + game auto-discovery + sanitizer/LTO/native options
├── engine/
│   ├── core/api.hpp            GameLike concept + conventions (READ THIS to write a game)
│   ├── core/rng.hpp            splitmix64 RNG, mix64 hash helper
│   ├── mcts/mcts.hpp           PUCT MCTS (split select/expand/backup)
│   ├── mcts/adaptive.hpp       min/target/max sims, early stop, extend-when-unstable
│   ├── inference/evaluator.hpp Evaluator interface (batched) + Uniform + Rollout evaluators
│   ├── selfplay/selfplay.hpp   self-play games + binary sample file format v2
│   ├── evaluation/elo.hpp      Elo + 95% CI from W/L/D
│   ├── time_manager/           gameplay clock allocation
│   └── training/               (empty — training lives in Python)
├── games/
│   ├── template/               Take-away (copy me to start a new game)
│   └── connect4/               bitboard game.hpp + reference_game.hpp
├── tools/                      validate_game, benchmark_game, selfplay, evaluate, play, common.hpp
├── python/gai/                 data.py, model.py, train.py, orchestrator.py  (untested)
├── configs/                    competition_60m.yaml, long_term.yaml
├── tests/                      test_time_manager.cpp, test_mcts.cpp (+ validate_<game> via CTest)
└── docs/                       HANDOFF.md, ADDING_A_GAME.md, REUSE.md
```

## 5. Component-by-component status

| Component | Status | Notes |
|---|---|---|
| Game API (`GameLike`) | ✅ Done, tested | Static-function struct per game; `Move` value == policy index |
| CMake + auto game registry | ✅ Done, tested | Globs `games/*/game.hpp`, generates `games.inc`; namespace must be `game_<dirname>` |
| Template game (Take-away) | ✅ Done, tested | Has reference impl; MCTS solves it (first move takes 1) |
| Connect-4 | ✅ Done, tested | Bitboard, reference, undo, mirror symmetry; 3000-game validation passes |
| `validate_game` | ✅ Done, tested | Random playouts, differential vs reference, undo, symmetry, zero-sum, encode determinism, state corruption, game-length bound |
| `benchmark_game` | ✅ Done, tested | One JSON line: speeds of legal/apply/terminal/encode/hash/playouts/MCTS |
| MCTS | ✅ Done, tested | PUCT, virtual loss, batched leaves, Dirichlet, temperature, FPU, tree reuse, node compaction |
| Adaptive search | ✅ Basic version | Early stop when leader can't be overtaken; extend when unstable/contested |
| TimeManager | ✅ Done, unit-tested | soft/hard limits, increment, safety margin, emergency mode; min-think can never cause a timeout |
| Self-play | ✅ Done, tested | Multi-threaded independent games; sample file v2 (planes/h/w/actions header) |
| `evaluate` (arena + Elo) | ✅ Works with uniform/rollout evaluators only | `--model` is rejected until NN evaluator exists |
| `play` | ✅ Works with uniform/rollout evaluators only | human/random/mcts, simulated `--tc BASE+INC` clock |
| Python models/train/orchestrator | ⚠️ Written, **never run** | No torch available; syntax-checked only |
| C++ NN evaluator | ❌ Not started | **Critical path** |
| GPU batching / CUDA | ❌ Not started | |
| Transposition table / NN cache | ❌ Not started | |
| Symmetry augmentation in training | ❌ Not started | Needs symmetry tables exported from C++ |
| Arena with NN, champion promotion | ❌ Not started | |
| Configuration racing, game profiler → config | ❌ Not started | Schema stub in `competition_60m.yaml` |
| `./build/train` CLI | ❌ Not started | Should wrap `python -m gai.orchestrator` |
| Long-term mode loop / experiments | ❌ Not started | Config exists |
| Gumbel / Sequential Halving / MuZero / EfficientZero | ❌ Not started | Phase 8 only |
| PGO, SIMD tuning, lock-free queues | ❌ Not started | Only if profiling justifies |
| Learned time management + gameplay data logging | ❌ Not started | Optional per the brief |
| TSan run | ⚠️ Option exists, not run | |

## 6. What was verified (evidence, not claims)

- `ctest`: 4/4 pass (time manager, MCTS, validate_connect4, validate_template) — in a normal build, an ASan+UBSan build, and a clean build from the zip.
- **Mutation test:** weakening the diagonal win check in a scratch copy made `validate_game` fail ("terminal status differs from reference"). The differential test works.
- MCTS finds an immediate Connect-4 win, blocks an immediate loss, and solves Take-away, with batch sizes 1 and 8.
- Python `read_samples` parsed real self-play output: shapes `(573,2,6,7)`, policies sum to 1, values in {-1,0,1}.
- Arena smoke test (rollout evaluator): 800 sims vs 100 sims won 18–2 over 20 games (CI is huge at that sample size).
- Timed play: MCTS adapts per move and shows `[early]` stops; no timeouts.

**Not verified:** any Elo improvement from *training*. No network has ever been trained or used in this repo.

## 7. Reference numbers (single shared vCPU — do not trust for planning)

Connect-4 `benchmark_game`: legal_moves ≈ 8.6e7/s · apply+hash ≈ 6.5e7/s · encode ≈ 8.7e6/s ·
random games ≈ 4.2e6/s · MCTS uniform-eval ≈ 4e6 sims/s · MCTS 1-rollout/leaf ≈ 1.9e6 sims/s ·
self-play (rollout, 200 sims, 1 thread) ≈ 850 games/s. With a real NN, inference will dominate and these become irrelevant.

## 8. "Ideas we stole" — honest provenance

**No code was copied from any other project.** Everything in the repo was written from scratch in this session.
The *ideas* come from general knowledge of the field, and I could not open the upstream repositories to verify
details, licenses, or performance claims. Treat this section as intellectual lineage, not verified reuse.

| Idea | Origin (from general knowledge) | Where it appears here |
|---|---|---|
| Self-play + MCTS + policy/value net training loop | AlphaGo Zero / AlphaZero (DeepMind) | Overall architecture, sample format (state, visit-policy, outcome) |
| PUCT selection, Dirichlet root noise, temperature for early moves | AlphaZero | `engine/mcts/mcts.hpp` |
| Virtual loss for parallel/batched leaf selection | Standard parallel-MCTS technique | `select_leaf`/`finish_leaf`, `run(..., batch)` |
| First-play urgency (FPU) reduction | Leela Zero / KataGo-style engines | `fpu_reduction` in `pick_child` |
| Tree reuse between moves | Common in Go/chess engines | `advance_root`, `compact()` |
| Bitboard Connect-4 (7 bits/column with sentinel, shift-and-mask 4-in-row test, `mask + bottom` drop) | John Tromp's well-known Connect-4 bitboard technique | `games/connect4/game.hpp` |
| Zobrist/mixing-style hashing | General game-engine practice (we used a splitmix64 mixer, not true Zobrist) | `hash()` in games |
| Reference-vs-optimized differential testing | General software-verification practice | `validate_game`, `reference_game.hpp` |
| Batched inference queue between search workers and GPU | Used by Leela Zero, KataGo, Minigo-style systems | Designed (interface only); implementation TODO |
| Evaluator abstraction decoupling search from the network | Common in AlphaZero implementations (alpha-zero-general style) | `Evaluator<G>` |
| Sequential Halving / Gumbel root selection | Gumbel AlphaZero / `mctx` (DeepMind) | Not implemented; planned for Phase 8 |
| Configuration racing (successive halving of candidates) | Hyperparameter-search practice (Hyperband/successive halving) | Config stub only |
| Elo from W/L/D with normal-approx CI | Standard chess-engine testing practice | `engine/evaluation/elo.hpp` |
| Simulated chess-style clocks with increment | Chess-engine time management | `TimeManager`, `play --tc` |

Projects named in the brief for the next agent to *study* (unverified, see `docs/REUSE.md`): OpenSpiel,
alpha-zero-general, LightZero, Minigo, KataGo, Leela Zero, mctx. Check each LICENSE file before copying any code.

## 9. Design decisions worth knowing

- **Compile-time dispatch, no virtual game interface.** Tools are templates dispatched by game name through a CMake-generated X-macro list. Cost: longer compile; benefit: inlined hot loops.
- **Header-only games.** The brief suggested `game.cpp / encoding.cpp / symmetry.cpp / tests.cpp`; we use `game.hpp` + `reference_game.hpp` for inlining and simplicity.
- **Evaluator priors are aligned to the legal-move list**, not the action space (keeps large action spaces cheap).
- **Value convention:** `Node::w` is stored from the view of the player who chose the incoming edge, so parents read child Q directly.
- **Two players, zero-sum, ids 0/1.** Non-alternating turns work in MCTS; single-player and >2-player games would need engine changes.
- **State must be small and trivially copyable** (MCTS copies it every simulation).
- **Python for orchestration/training only**; no MCTS in Python.
- **Per-game `config.yaml` is documentation only** for now (merge logic is TODO).

## 10. Known limitations / simplifications

- No transposition table; no NN evaluation cache.
- `evaluate` uses fixed random openings, not paired-color openings; 20–100 game matches have wide CIs.
- Adaptive-search thresholds (0.5 stability window, 0.6 runner-up ratio) are hand-picked, not tuned.
- `apply()` with an illegal move is undefined and not tested.
- Self-play threads each own a tree; there is no cross-thread batching.
- Sample files are raw float32 — fine for 60-minute runs, wasteful for long-term runs.
- `-march=native` is on by default (not portable; use `-DGAI_NATIVE=OFF` for shared binaries).
- The validator checks symmetry only for games that declare `kNumSymmetries > 1` and assumes a symmetric start position.
- GitHub inspection of upstream projects was not possible; REUSE.md licenses are unverified.

## 11. What needs to be done — prioritized plan for the next agent

### Step 0 — Set up a real machine
GPU box with CUDA, `pip install -r python/requirements.txt`, then `python -m gai.model` (expect `mlp ok`, `resnet ok`).
Fix whatever breaks: all PyTorch code is untested.

### Step 1 — C++ NN evaluator (critical path)
- Implement `NNEvaluator<G> : Evaluator<G>` in `engine/inference/`.
- Contract is final: input is a batch of states; output priors (softmax over legal moves, in `legal_moves` order) and value (view of player to move).
- Fastest route: libtorch loading TorchScript from `python/gai/train.py:export_torchscript`. Gate behind a CMake option (e.g. `GAI_LIBTORCH`) and keep the CPU-only build working.
- Wire `nn:<path>` into `make_evaluator` (`tools/common.hpp`); enable `--model` in `evaluate` and `play`.
- Only consider TensorRT / CUDA graphs / ONNX if profiling shows end-to-end gain.

### Step 2 — Cross-thread dynamic batching
Shared inference queue: workers enqueue (state, completion); a GPU thread forms batches (max size / max wait µs), uses pinned memory and a CUDA stream. `select_leaf`/`finish_leaf` already support asynchronous use.

### Step 3 — Close the training loop
Orchestrator switches self-play to the latest NN after iteration 1; arena new checkpoint vs champion; promote on score threshold; write `models/champion.pt`; log Elo vs wall-clock. **First milestone to prove: Connect-4 Elo rising with training time.**

### Step 4 — Make it fast for real
Profile (`benchmark_game`, `nvidia-smi`, `perf`). Then, in order of expected payoff: NN cache keyed by `G::hash`, transposition table, symmetry augmentation (export symmetry permutations from C++), playout-cap randomization, mixed precision tuning, worker-count auto-selection, CPU affinity. Benchmark every change as **Elo/minute**, not sims/s.

### Step 5 — Competition-mode intelligence
Game profiler → automatic config selection (network size, sims, batch size); configuration racing (early: several configs, middle: eliminate, late: commit); `./build/train` wrapper; per-game `config.yaml` merge; checkpoint-vs-champion Elo tracking; logging of git commit, hardware, config, losses, GPU/CPU utilization as JSONL.

### Step 6 — Long-term mode
Continuous self-play/train/eval loop, champion promotion, sharded compressed replay storage, experiment runner (network size, c_puct, sims, replay window, LR schedule).

### Step 7 — Phase 8 (only after Steps 1–4 work)
Gumbel AlphaZero / Sequential Halving, MuZero/EfficientZero evaluation, learned time management from logged gameplay data, PGO/SIMD, TSan runs, optional WASM target (never at the cost of native speed).

## 12. Quick-start commands

```bash
pip install cmake                                   # if cmake is missing
cmake -S . -B build && cmake --build build -j && (cd build && ctest)

./build/validate_game --game connect4               # contract + differential tests
./build/benchmark_game --game connect4              # JSON profile
./build/selfplay --game connect4 --games 100 --sims 200 --out /tmp/d.bin
./build/evaluate --game connect4 --games 50 --sims-a 800 --sims-b 100
./build/play --game connect4 --p0 human --p1 mcts --tc 60+1

# sanitizers
cmake -S . -B build-asan -DGAI_SANITIZE="address;undefined" -DCMAKE_BUILD_TYPE=Debug -DGAI_NATIVE=OFF

# Python pipeline (UNTESTED)
cd python && python -m gai.orchestrator --config ../configs/competition_60m.yaml --game connect4 --minutes 2
```

## 13. How to add a game (summary)

1. `cp -r games/template games/<name>`; rename namespace `game_template` → `game_<name>`.
2. Write `GAME_RULES.md`.
3. Implement `game.hpp` (state, constants, moves, outcome, encode, hash, optional undo/symmetries) — don't touch `engine/`.
4. Implement a slow, independently-written `reference_game.hpp`.
5. Re-run `cmake`, then `./build/validate_game --game <name>` until it prints `VALIDATION PASSED`; run once under ASan/UBSan.
6. `./build/benchmark_game --game <name>`; optimize hot paths and re-validate.
7. Train.

Common validator failures: moves outside `[0, kActionCount)`, `kMaxMoves` too small, non-zero-sum outcome,
`encode` not from the mover's view, forgetting to switch players, games exceeding `kMaxGameLength`.

## 14. Risks to keep in mind

- Biggest risk: building lots of infrastructure before proving that training improves Elo. Do Steps 1–3 on Connect-4 first.
- Raw throughput ≠ strength. A faster but weaker search config is a regression.
- Untested Python/GPU code probably has bugs; budget time to debug it.
- Unverified licenses: do not copy upstream code without reading the LICENSE.
- Games with large state or huge action spaces may need engine changes (state stored in nodes, sparse policies).
