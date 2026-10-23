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

## 2026-10-21: Elo-methodology fix batch — partial, finish here

Three GLM-5.3 subagents dispatched (equal-time MCTS + Dirichlet auto + 2-ply tactics; Python SPRT/replay/plumbing; Connect4 solver + alpha-beta agent). Two completed, one aborted mid-task. Nothing committed yet — all work is uncommitted working-tree changes. `bots/meridian/*` and `docs/meridian/TRAINING_FAST.md` changes are SEPARATE user work, do not touch.

### DONE (completed subagents)
- **Equal-time arenas** (`engine/mcts/mcts.hpp`, `tools/evaluate.cpp`): `--tc-ms N` (equal clock both agents), `--tc-a/--tc-b` (race), measured `ms_move_*_mean/p95` in JSON. `MCTS::set_deadline()` checked per batch after ≥1 sim. Sims-only behavior repro'd exactly.
- **Dirichlet auto** (`mcts.hpp`, `tools/selfplay.cpp`): `--dirichlet <= 0` now means AUTO = 10/#root_children per root. NOTE the semantic change: 0 is no longer "noise off"; do it programmatically via `dirichlet_alpha=0, dirichlet_auto=false`.
- **2-ply tactics** (`engine/selfplay/selfplay.hpp`): `find_forced_win_2ply<G>()`, `--tactics2 1` (default OFF), `GameRecord::tactic2_solves`. Verified vs depth-3 negamax: 2794/2794 agreement.
- **Python** (`orchestrator.py`, `data.py`, `pick_champion.py`, `sprt.py`): SPRT gate budget-clamped and reserved inside the hour (new keys `eval.sprt_elo1` default 35.0, `gate_share` 0.1, `gate_minutes_per_game`, `gate_time_safety`); equal-time eval plumbing (`eval.tc_ms/tc_a/tc_b`, ≥20 ms); `eval.ordered` + `eval.roundrobin_last` Bradley-Terry endgame (`bradley_terry_elo()` in sprt.py); `ReplayBuffer.add` vectorised (bit-identical); new `configs/connect4_60m_v9.yaml`; `--dry-run` timeline flag. `pytest python/tests -q` → 46 passed. `solve_oracle` + new files compile clean.

### Build note
`cmake` is not on PATH in this sandbox. Use `pip install cmake` or the absolute path `$HOME/.local/lib/python3.10/site-packages/cmake/data/bin/cmake`.
### Verify commands
  - `./build/evaluate --game connect4 --games 8 --tc-ms 20`
  - `./build/selfplay --game connect4 --games 4 --sims 32 --tactics2 1`
  - `./build/solve_oracle --game connect4 --solve-initial`
  - `./build/solve_oracle --game connect4 --positions 200 --out /tmp/opencode/labels_test.bin`
  - `./build/validate_game --game connect4`
  - `python -m pytest python/tests -q`
- Solver compartment: engine/evaluation/{solver.hpp, ab_agent.hpp}, tools/solve_oracle.cpp already created, CMake target builds.

### TODO to finish (resume subagent C or take over hand-crafted):
1. **Verify solver.hpp exactness**: `--solve-initial` must report P0 win (center), print nodes/seconds; deadline: <60 s (optimize ordering / bitboard TT if >120 s).
2. **Oracle labels**: `--positions 200 --out /tmp/opencode/labels_test.bin` must run <3 min and write the binary layout {hash, value_from_p0, best_move}; then wire the labels into checkpoint eval as an optimal-move-rate gate (next agent's choice).
3. **Wire ab_agent into arena**: `evaluate.cpp` does NOT have an `ab:` agent type yet; add `--evaluator-a ab:depth=D:ms=M` (select_move per move at equal `--tc-ms`), consistent JSON key on names for winner attribution.
4. **Race agent types**: new config `race.enabled 1` candidates now: `[rollout]`, `[nn]`, `[ab:...]` at equal tc; orchestrator must allow agent-type candidates, not just architectures.
5. **Configs**: refresh `configs/connect4_60m_v9.yaml` with `dirichlet: 0` (auto), `eval.tc_ms: 30`, `eval.ordered: true`, `tactics2` A/B flag, race enabled. Also update `docs/REUSE.md` probe note about `--dirichlet 0` semantic change.
6. **Docs**: update README + docs/TODO.md status tables to reflect shipped items above (equal-time, SPRT 35, vectorised replay, dry-run, solver files present).
7. **Tests**: add C++ tests for `find_forced_win_2ply` and deadline-API; keep existing green.
8. **Known caveats from agent A**: policy-target floats differ ~1 ULP from pristine binary in prune=1 path (compiler FP-contraction, not logic); p95 slightly > tc due to chunk granularity; `--sims` default when tc-only raises to 2e6 cap.

# JAM PLAN — Strat Jam hard-mode: 1h training → 20min game, TC 2+15

Format: train ~50-55 min, then a live 20-min game at 2+15 (≈20-42 plies → ~10 min total think
time available → serving sims can be WORLD-CLASS: 600-1200 sims/move is affordable).

## Artifacts plan (already supported, no new code needed)
- `python -m gai.orchestrator --config configs/connect4_60m_v9.yaml --game connect4 --minutes 50`
- Every iteration exports `latest.ts` and the next iter plays with `hybrid:latest_ts` (annealed):
  the live bot improves continuously through the hour. NO restarts needed.
- `checkpoint.interval_minutes: 5` → candidate every 5 min (last 5 kept).
- Endgame: `eval.ordered true` + `roundrobin_last 5` → Bradley-Terry picks the strongest
  checkpoint automatically in the reserved gate_share window. That checkpoint(esTorchScript)
  is the one that plays the live game.
- `--dry-run` first to rehearse the full timeline (no training side effects).

## Game-phase serving recipe (measured basis)
- Single champion net, argmax pick, generous max_sims (600-1200): suite shows +140-170 Elo
  per sims-doubling in 40-320 range; never tested above 320 — measure tonight (ruler).
- `ensemble:a+b+c` at FIXED sims = +89 Elo (2.9× cost). At 15s/move increment you can afford
  it — re-measure ensemble vs single at equal WALL-CLOCK (tc-ms 1000+) tonight before choosing.
- LCB pick: neutral at fast TC; test at high sims; keep only on suite win.
- Early stops are already banked (proven-win stop / can't-be-overtaken / shortcuts), so
  effective sims/move stays high where it matters (midgame). 1s/move safety margin in clocks.
- Adaptive per-move denken: the engine already thinks longer on contested positions
  (adaptive extend-when-unstable) and instantly plays proven wins/forced shorts. Do NOT add
  NEW time logic without a suite A/B — recent kill history says knob-level changes mostly
  fail (Gumbel ×3, safety_factor, LCB@fast).

## Pondering (think during opponent's clock) — OPTIONAL, only if flag-safe
- Engine has no ponder yet. Cheap version alive as TODO: background thread that pre-searches
  the opponent's most likely replies (from the reused root's visit distribution) and hands the
  tree over on our turn; tree-reuse advance_root already exists. ONLY implement behind
  `--ponder 1` with a suite A/B; if it loses, drop it — do not jam-day-without-a-test.

## Speed so the 1 training-hour is real
- GPU: install cu126 torch (CPU-only wheel was 2.13.0+cpu), rebuild build-lt against it,
  `GAI_NN_DEVICE=cuda` set for selfplay/evaluate. Without GPU everything else is marginal.
- Already fast: lanes batched inference (2.11×), batch leaf dedup, hash-keyed eval cache,
  forced playouts + target pruning, can't-be-overtaken stops, tree reuse.
- Training flags for jam: `training.num_threads 2` (ResNet) / 1 (MLP), `dup_weight` ON
  (trend +49..+79 @50-100 sims, unconfirmed at 60m — acceptable risk given format),
  resign probe `resign_q: -0.90` + `resign_probe: 0.15` (verify false_resign_rate < ~5% from
  the run JSON first), multi knicker knuckles per round of changes = one A/B per flag only.

## Opening
- P0 book: center column (solved-game knowledge). Verify `./build/solve_oracle --game
  connect4 --solve-initial` runtime < 60 s (was a pending TODO); if it solves faster,
  consider a 2-4 ply exact book; otherwise center-only + full search.

## Rehearsal (do all of these before the jam)
1. `pip install torch --index-url .../cu126` + rebuild build-lt + GAI_NN_DEVICE=cuda →
   `./build-lt/evaluate nn:...` loads on cuda (speed printed in JSON).
2. Suite ruler at fixed 600 / 1000 sims (single vs ensemble vs LCB) — pick serving config
   by measured Elo, not vibes.
3. 15-min training pilot with the final jam flags → check positions/s, steps/iter ≥ 10,
   false_resign_rate, checkpoint + export no failures.
4. `--dry-run` full 50-min timeline.
5. `play --p1 mcts --tc 120+15` simulation to sanity-check clock behavior incl. 1s margin.

## Subagents for this task (MANDATORY: run via background subagents, MAX 3, GLM-5.3 nvidia)
- A/B/C role split: keep working-tree changes non-overlapping (one agent owns engine/**,
  one owns tools/**, one owns configs/**+python/**). ctest + pytest must stay green.
  Every change: measure first (ruler/suite A/B), kill on no win (protocol: docs/TODO.md).

## 2026-10-23 measured: lanes plateau + GPU verdict (jam-critical)
- Selfplay throughput sweep, connect4, nn:models/latest.ts, 2 threads, 600 games:
  lanes 8/batch 16 (v9) = 56.6 gps → **lanes 96-128 / batch 48 = ~180 gps (3.2×)**.
  Plateau: lanes 96/128 both ≈180; batch 48 > 64 within noise. Any jam run uses
  `--lanes 96 --batch 48` (or via yaml selfplay.lanes 96 / batch 48).
- CUDA torch 2.14.1+cu126 installed (cuda available True). Existing build-lt binary
  loads it fine (soname compat). GPU TRAINING measured 7.7× faster (91 vs 12 steps/s
  for resnet32 batch 256). GPU INFERENCE is 10× SLOWER on GTX1650 4GB at connect4 net sizes
  (evaluate 8× slower; selfplay 6.5 gps vs 180): do NOT set GAI_NN_DEVICE=cuda for
  selfplay/serve — CPU inference through big lane batches wins outright.
- Jam winner combination: CPU lanes96 + GPU resnet training pass.
