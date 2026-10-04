// Evaluator interface: the ONLY thing MCTS needs from a neural net (or heuristic).
// Batched from the start so a GPU evaluator can slot in without touching MCTS.
//  priors[i] corresponds to the i-th move returned by G::legal_moves(state) and sums to 1.
//  value is from the view of G::current_player(state), in [-1, 1].
#pragma once
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"

namespace gai {
template <GameLike G> struct EvalResult {
  float value = 0;
  int n = 0;
  std::array<float, G::kMaxMoves> priors{};
};

template <GameLike G> struct Evaluator {
  virtual ~Evaluator() = default;
  virtual void evaluate(const typename G::State* states, EvalResult<G>* out, int count) = 0;
};

// Uniform priors, value 0. Baseline / sanity evaluator.
template <GameLike G> struct UniformEvaluator : Evaluator<G> {
  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    Move mv[G::kMaxMoves];
    for (int i = 0; i < count; i++) {
      int n = G::legal_moves(states[i], mv);
      out[i].n = n; out[i].value = 0;
      for (int j = 0; j < n; j++) out[i].priors[j] = 1.0f / n;
    }
  }
};

// Uniform priors, value = mean outcome of random playouts. Surprisingly strong baseline for tests.
template <GameLike G> struct RolloutEvaluator : Evaluator<G> {
  int rollouts; Rng rng;
  explicit RolloutEvaluator(int r = 1, uint64_t seed = 1) : rollouts(r), rng(seed) {}
  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    Move mv[G::kMaxMoves];
    for (int i = 0; i < count; i++) {
      int n = G::legal_moves(states[i], mv);
      out[i].n = n;
      for (int j = 0; j < n; j++) out[i].priors[j] = 1.0f / n;
      int me = G::current_player(states[i]); float sum = 0;
      for (int r = 0; r < rollouts; r++) {
        auto s = states[i];
        for (int ply = 0; ply < G::kMaxGameLength && !G::is_terminal(s); ply++) {
          int k = G::legal_moves(s, mv);
          G::apply(s, mv[rng.below(k)]);
        }
        sum += G::outcome(s, me);
      }
      out[i].value = sum / rollouts;
    }
  }
};

// NN evaluation cache keyed by G::hash (transpositions + tree reuse). Wraps any evaluator.
template <GameLike G> struct CachingEvaluator : Evaluator<G> {
  explicit CachingEvaluator(std::unique_ptr<Evaluator<G>> inner, size_t cap = 1 << 16)
      : inner_(std::move(inner)), cap_(cap) {}
  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    std::vector<int> missing;
    missing.reserve(count);
    for (int i = 0; i < count; i++) {
      auto it = table_.find(G::hash(states[i]));
      if (it != table_.end()) { out[i] = it->second; hits++; }
      else missing.push_back(i);
    }
    if (!missing.empty()) {
      std::vector<typename G::State> ms(missing.size());
      std::vector<EvalResult<G>> mr(missing.size());
      for (size_t k = 0; k < missing.size(); k++) ms[k] = states[missing[k]];
      inner_->evaluate(ms.data(), mr.data(), (int)ms.size());
      for (size_t k = 0; k < missing.size(); k++) {
        out[missing[k]] = mr[k];
        misses++;
        if (table_.size() < cap_) table_.emplace(G::hash(ms[k]), mr[k]);
      }
    }
  }
  long hits = 0, misses = 0;

 private:
  std::unique_ptr<Evaluator<G>> inner_;
  size_t cap_;
  std::unordered_map<Hash, EvalResult<G>> table_;
};

// TODO(next agent): NNEvaluator (libtorch/TensorRT) with a dynamic-batching queue shared by MCTS workers.
// See docs/HANDOFF.md section "GPU inference".
}  // namespace gai
