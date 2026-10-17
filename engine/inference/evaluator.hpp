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

// Checkpoint ensemble: mean value + averaged priors (renormalized) over N inner evaluators.
// Serve-side variance reduction (last-k checkpoints) for zero retraining cost.
// Members MUST be distinct model instances with no shared mutable state: each is evaluated
// independently on every call. Priors are averaged in legal-move-index space, which is
// consistent across members because they all evaluate the same state and G::legal_moves
// is deterministic for a given state.
template <GameLike G> struct EnsembleEvaluator : Evaluator<G> {
  explicit EnsembleEvaluator(std::vector<std::unique_ptr<Evaluator<G>>> members)
      : members_(std::move(members)) {}
  void evaluate(const typename G::State* states, EvalResult<G>* out, int count) override {
    if (members_.empty()) { uniform(states, out, count); return; }
    std::vector<EvalResult<G>> acc(count), tmp(count);
    for (auto& m : members_) {
      m->evaluate(states, tmp.data(), count);
      for (int i = 0; i < count; i++) {
        int n = tmp[i].n < 0 ? 0 : (tmp[i].n > G::kMaxMoves ? G::kMaxMoves : tmp[i].n);
        acc[i].n = n;
        acc[i].value += tmp[i].value;
        for (int j = 0; j < n; j++) acc[i].priors[j] += tmp[i].priors[j];
      }
    }
    const float inv = 1.f / (float)members_.size();
    for (int i = 0; i < count; i++) {
      out[i].n = acc[i].n;
      out[i].value = acc[i].value * inv;
      float s = 0;
      for (int j = 0; j < acc[i].n; j++) { acc[i].priors[j] *= inv; s += acc[i].priors[j]; }
      if (s > 0) { for (int j = 0; j < acc[i].n; j++) out[i].priors[j] = acc[i].priors[j] / s; }
      else { for (int j = 0; j < acc[i].n; j++) out[i].priors[j] = acc[i].n ? 1.f / (float)acc[i].n : 0.f; }
    }
  }
  size_t size() const { return members_.size(); }

 private:
  void uniform(const typename G::State* states, EvalResult<G>* out, int count) {
    Move mv[G::kMaxMoves];
    for (int i = 0; i < count; i++) {
      int n = G::legal_moves(states[i], mv);
      out[i].n = n; out[i].value = 0;
      for (int j = 0; j < n; j++) out[i].priors[j] = n ? 1.f / (float)n : 0.f;
    }
  }
  std::vector<std::unique_ptr<Evaluator<G>>> members_;
};

// TODO(next agent): NNEvaluator (libtorch/TensorRT) with a dynamic-batching queue shared by MCTS workers.
// See docs/HANDOFF.md section "GPU inference".
}  // namespace gai
