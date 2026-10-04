// Adaptive-budget search: min/target/max simulations + soft/hard time. Deterministic policy v1:
//  - stop early when the leader cannot be overtaken within the target budget
//  - at target/soft deadline: stop if stable, otherwise extend towards max_sims / hard deadline
#pragma once
#include <chrono>
#include "engine/mcts/mcts.hpp"

namespace gai {
struct SearchLimits {
  int min_simulations = 200, target_simulations = 2000, max_simulations = 20000;
  double soft_ms = 1e18, hard_ms = 1e18;  // 1e18 = no time limit (simulation-count mode)
  int batch = 1;
  int check_every = 16;
  double safety_factor = 1.0;  // >1 prunes earlier, 0 disables can't-be-overtaken stop
};
struct SearchReport {
  Move best = -1; int sims = 0; double ms = 0; float root_value = 0, entropy = 0;
  bool early_stop = false, extended = false, hit_hard = false;
};

template <GameLike G>
SearchReport adaptive_search(MCTS<G>& mcts, Evaluator<G>& ev, const SearchLimits& L) {
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now();
  auto ms = [&] { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); };
  SearchReport r;
  int start = mcts.total_visits();
  Move leader = -1; int last_change = 0; int sims = 0;
  while (true) {
    mcts.run(L.check_every, ev, L.batch);
    sims = mcts.total_visits() - start; double t = ms();
    if (G::is_terminal(mcts.root_state())) break;
    int b, s2; mcts.top2_visits(b, s2);
    Move cur = mcts.best_move();
    if (cur != leader) { leader = cur; last_change = sims; }
    if (t >= L.hard_ms) { r.hit_hard = true; break; }
    if (sims >= L.max_simulations) break;
    if (sims < L.min_simulations) continue;
    if (mcts.root_proven() == 1) { r.early_stop = true; break; }  // proven win: play it now
    bool stable = (sims - last_change) >= 0.5 * sims;          // leader unchanged for last half of search
    bool contested = s2 > 0.6 * b;                              // runner-up close
    bool uncertain = !stable || contested;
    // Can't-be-overtaken vs real remaining budget (max_sims), with safety factor.
    if (L.safety_factor > 0) {
      int remaining = L.max_simulations - sims;
      if (remaining < 0) remaining = 0;
      double need = (double)(b - s2) * L.safety_factor;
      if (need > (double)remaining + (double)L.batch) { r.early_stop = true; break; }
    } else if (b - s2 > L.target_simulations - sims && sims < L.target_simulations) { r.early_stop = true; break; }
    if (sims >= L.target_simulations || t >= L.soft_ms) {
      if (!uncertain) break;
      r.extended = true;  // keep going towards max_simulations / hard_ms
    }
  }
  r.best = mcts.best_move(); r.sims = sims; r.ms = ms();
  r.root_value = mcts.root_value(); r.entropy = mcts.policy_entropy();
  return r;
}
}  // namespace gai
