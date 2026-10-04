// HybridEvaluator: annealed handover from rollout leaf values to NN leaf values.
// value = alpha * v_nn + (1 - alpha) * v_rollout
// priors = alpha * p_nn + (1 - alpha) * uniform
// alpha = 0  -> pure rollout teacher (proven quality, use while the net is weak)
// alpha = 1  -> pure NN (identical to nn:<path>)
// The orchestrator ramps alpha over nn_anneal_iters starting at nn_start_iter.
// Requires GAI_LIBTORCH (wraps NNEvaluator); rollout fallback is always available.
#pragma once

#include <string>
#include "engine/inference/evaluator.hpp"
#include "engine/inference/nn_evaluator.hpp"

namespace gai {

template <GameLike G> struct HybridEvaluator : Evaluator<G> {
  HybridEvaluator(std::string path, float alpha, int rollouts = 1, uint64_t seed = 1)
      : nn_(std::move(path)), roll_(rollouts, seed), alpha_(alpha < 0.f ? 0.f : alpha > 1.f ? 1.f : alpha) {}

  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    if (alpha_ <= 0.f) { roll_.evaluate(states, out, count); return; }
    std::vector<EvalResult<G>> rn(count), rr(count);
    nn_.evaluate(states, rn.data(), count);
    if (alpha_ >= 1.f) {
      for (int i = 0; i < count; i++) out[i] = rn[i];
      return;
    }
    roll_.evaluate(states, rr.data(), count);
    Move mv[G::kMaxMoves];
    for (int i = 0; i < count; i++) {
      int n = G::legal_moves(states[i], mv);
      out[i].n = n;
      out[i].value = alpha_ * rn[i].value + (1.f - alpha_) * rr[i].value;
      for (int j = 0; j < n; j++)
        out[i].priors[j] = alpha_ * rn[i].priors[j] + (1.f - alpha_) * (n ? 1.f / (float)n : 0.f);
    }
  }

 private:
  NNEvaluator<G> nn_;
  RolloutEvaluator<G> roll_;
  float alpha_;
};

}  // namespace gai
