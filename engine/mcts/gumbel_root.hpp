// Gumbel root + Sequential Halving (Danihelka et al., ICLR 2022,
// "Policy Improvement by Planning with Gumbel").
//
// Math reference: the paper + DeepMind's mctx qtransform/sigma (Apache-2.0).
// Verify constants against mctx before trusting serve-time behavior.
// Stage-1 gate (no training): improved-policy top-1 accuracy vs visit-count
// targets on solver-labelled positions at n = 8..128.
//
// Replaces (do NOT combine with): Dirichlet root noise, forced playouts,
// temperature exploration. The Gumbel sample IS the exploration, and the
// improved-policy target stays valid at 16-32 sims. Deterministic (no-Gumbel)
// selection when serving.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "engine/core/rng.hpp"
#include "engine/mcts/mcts.hpp"

namespace gai {

static inline float sample_gumbel(Rng& rng) {
  float u = rng.uniform();
  if (u <= 0.f) u = 1e-7f;
  if (u >= 1.f) u = 1.f - 1e-7f;
  return -std::log(-std::log(u));
}

template <GameLike G> struct GumbelResult {
  Move best = -1;
  std::vector<float> improved;  // size G::kActionCount, sums to 1
};

// Gumbel-Top-m (m = all root children) + Sequential Halving over cfg.gumbel_sims.
// Returns the SH winner and the improved policy softmax(logits + sigma(completedQ)).
template <GameLike G>
GumbelResult<G> run_gumbel_root(MCTS<G>& mcts, Evaluator<G>& ev, const MctsConfig& cfg, Rng& rng, int batch = 1) {
  GumbelResult<G> out;
  out.improved.assign(G::kActionCount, 0.f);
  if (!mcts.root_expanded()) mcts.run(1, ev, 1);
  int nc = mcts.num_root_children();
  if (nc <= 0) return out;
  if (nc == 1 || mcts.root_proven() == 1) {
    out.best = mcts.best_move();
    if (out.best >= 0 && out.best < G::kActionCount) out.improved[out.best] = 1.f;
    return out;
  }
  // Logits = log-softmax over legal-move priors.
  std::vector<float> logits(nc);
  float mx = -1e30f;
  for (int i = 0; i < nc; i++) { logits[i] = std::log(std::max(mcts.child_prior(i), 1e-9f)); mx = std::max(mx, logits[i]); }
  float s = 0;
  for (int i = 0; i < nc; i++) s += std::exp(logits[i] - mx);
  float lse = mx + std::log(std::max(s, 1e-30f));
  for (int i = 0; i < nc; i++) logits[i] -= lse;
  // One Gumbel sample per candidate (fixed for the whole search).
  std::vector<float> g(nc);
  for (int i = 0; i < nc; i++) g[i] = sample_gumbel(rng);

  auto q01 = [&](int i) {
    float q = mcts.child_visits(i) > 0 ? mcts.child_q(i) : mcts.root_value();
    float v = (q + 1.f) * 0.5f;
    return v < 0.f ? 0.f : v > 1.f ? 1.f : v;
  };
  // Sigma rescale fix: raw (c_visit+maxN)*q01 has spread ~50-80 while logits
  // (log-softmax) spread ~2-4, so raw sigma swamps logits+Gumbel ~10:1 and
  // targets collapse to near-one-hot on noisy Q. Rescale sigma to the logit
  // spread (centered): all three score terms stay comparable, as in mctx.
  float lo_min = *std::min_element(logits.begin(), logits.end());
  float lo_max = *std::max_element(logits.begin(), logits.end());
  float lo_spread = std::max(lo_max - lo_min, 1e-3f);
  auto sigma_of = [&](int i, int maxN) {
    float raw_n = (cfg.gumbel_c_visit + (float)maxN) * cfg.gumbel_c_scale;
    // min/max/mean of raw across children at this maxN (raw linear in q01)
    float qmin = 1e30f, qmax = -1e30f, qsum = 0.f;
    for (int j = 0; j < nc; j++) {
      float q = q01(j);
      qmin = std::min(qmin, q); qmax = std::max(qmax, q); qsum += q;
    }
    float raw_spread = std::max((qmax - qmin) * raw_n, 1e-6f);
    float raw_mean = (qsum / (float)nc) * raw_n;
    float raw_i = raw_n * q01(i);
    return (raw_i - raw_mean) * (lo_spread / raw_spread);
  };
  auto maxN = [&] {
    int m = 0;
    for (int i = 0; i < nc; i++) m = std::max(m, mcts.child_visits(i));
    return m;
  };
  auto score = [&](int i) { return g[i] + logits[i] + sigma_of(i, maxN()); };

  // Sequential Halving: log2(|cand|) phases, budget split evenly per phase.
  std::vector<int> cand(nc);
  std::iota(cand.begin(), cand.end(), 0);
  int phases = 0;
  for (int n = (int)cand.size(); n > 1; n = (n + 1) / 2) phases++;
  phases = std::max(phases, 1);
  int total = std::max(cfg.gumbel_sims, nc);  // at least one visit per candidate
  int used = 0, done_phases = 0;
  while (cand.size() > 1) {
    int left = phases - done_phases++;
    int share = std::max(1, (total - used) / std::max(left, 1) / std::max((int)cand.size(), 1));
    for (int idx : cand) {
      mcts.set_forced_root(idx);
      mcts.run(share, ev, batch);
      used += share;
    }
    mcts.clear_forced_root();
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return score(a) > score(b); });
    cand.resize(std::max((int)(cand.size() + 1) / 2, 1));
  }
  // Spend leftover budget on the winner (sharpens its completed Q).
  if (used < total && !cand.empty()) {
    mcts.set_forced_root(cand[0]);
    mcts.run(total - used, ev, batch);
    mcts.clear_forced_root();
  }
  // Final winner + improved policy over ALL legal moves.
  std::sort(cand.begin(), cand.end(), [&](int a, int b) { return score(a) > score(b); });
  int win = cand.empty() ? 0 : cand[0];
  // Re-score all children for the final best (Q changed during leftover spend).
  float bs = -1e30f;
  for (int i = 0; i < nc; i++) {
    float sc = score(i);
    if (sc > bs) { bs = sc; win = i; }
  }
  out.best = mcts.child_move(win);
  std::vector<float> z(nc);
  float zmx = -1e30f;
  for (int i = 0; i < nc; i++) {
    z[i] = logits[i] + sigma_of(i, maxN());
    zmx = std::max(zmx, z[i]);
  }
  float zsum = 0;
  for (int i = 0; i < nc; i++) zsum += std::exp(z[i] - zmx);
  for (int i = 0; i < nc; i++) out.improved[mcts.child_move(i)] = std::exp(z[i] - zmx) / std::max(zsum, 1e-30f);
  return out;
}

}  // namespace gai
