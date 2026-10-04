# Reuse table (NOT yet verified)
The authoring sandbox could not reach the GitHub API/network for repo inspection, so **licenses, performance claims and code quality below are unverified**. The "Reuse" column is a recommendation from general knowledge; the next agent must open each repo, confirm the LICENSE file and read the code before copying anything.

| Component | Project | Reuse | License | Benefit | Difficulty |
|---|---|---|---|---|---|
| Game API / algorithm reference | DeepMind OpenSpiel | Read for API ideas; optionally cross-check game rules | VERIFY | Many validated game implementations; AlphaZero reference | Low (reading) |
| AlphaZero training pipeline reference | alpha-zero-general (suragnair) | Read only (Python, slow) | VERIFY | Simple readable baseline | Low |
| Gumbel/MuZero/EfficientZero variants | LightZero (opendilab) | Study for Phase 8 | VERIFY | Many algorithm variants in one codebase | Medium |
| Batched self-play at scale | Minigo / KataGo / Leela Zero | Study architecture (inference queue, selfplay workers); consider KataGo's batching design | VERIFY | Proven large-scale C++ self-play + GPU batching designs | Medium–High |
| Gumbel MCTS in JAX | DeepMind mctx | Reference for Sequential Halving / Gumbel root selection math | VERIFY | Clean reference implementation | Low |
| NN inference | libtorch / TensorRT | Candidate runtime for NNEvaluator | VERIFY | Fast path to GPU inference | Medium |
