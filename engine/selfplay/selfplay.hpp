// Self-play game generation + binary sample file format (read by python/gai/data.py).
// File: magic "GAIS", then u32 x5: version(3), planes, height, width, action_count; then records of
//   float input[planes*h*w]  (encode() of the state, view of player to move)
//   float policy[action_count] (MCTS visit distribution; one-hot for tactical shortcuts)
//   float value               (final outcome from the view of the player to move, +1/0/-1)
//   float full                (1 = full-budget search, policy target valid; 0 = cheap/shortcut, value-only)
#pragma once
#include <cstdio>
#include <mutex>
#include <vector>
#include "engine/mcts/mcts.hpp"

namespace gai {
struct SelfPlayConfig {
  int simulations = 400;
  int batch = 1;
  int temperature_moves = 10;   // sample ~visits^(1/T) for the first N plies, then argmax
  float temperature = 1.0f;
  bool reuse_tree = true;
  bool enable_shortcuts = true;   // single-move skip + win-in-1 / block-in-1
  bool policy_pruning = true;     // KataGo-style: train on pruned target, play on raw visits
  float full_search_prob = 1.0f;  // playout-cap randomization: P(full sims); else cheap
  float cheap_sim_fraction = 0.125f;
  int cheap_min_sims = 10;
  MctsConfig mcts;
};

template <GameLike G> struct Sample {
  std::vector<float> input, policy; int player; float value = 0; float full_search = 1.f;
};
template <GameLike G> struct GameRecord { std::vector<Sample<G>> samples; float outcome_p0 = 0; int plies = 0; };

// Generic win-in-1: move that ends the game with a win for the mover.
template <GameLike G> bool find_win_in_1(const typename G::State& s, Move* out) {
  Move mv[G::kMaxMoves];
  int n = G::legal_moves(s, mv);
  int me = G::current_player(s);
  for (int i = 0; i < n; i++) {
    auto t = s;
    G::apply(t, mv[i]);
    if (G::is_terminal(t) && G::outcome(t, me) > 0) { *out = mv[i]; return true; }
  }
  return false;
}

// Generic block-in-1: exactly one of our moves leaves the opponent with no immediate win.
template <GameLike G> bool find_block_in_1(const typename G::State& s, Move* out) {
  Move mv[G::kMaxMoves];
  int n = G::legal_moves(s, mv);
  if (n <= 1) return false;
  Move win;
  int safe = 0;
  Move last_safe = -1;
  for (int i = 0; i < n; i++) {
    auto t = s;
    G::apply(t, mv[i]);
    if (G::is_terminal(t)) { safe++; last_safe = mv[i]; continue; }  // our own terminal: keep as candidate
    if (!find_win_in_1<G>(t, &win)) { safe++; last_safe = mv[i]; }
  }
  if (safe == 1) { *out = last_safe; return true; }
  return false;
}

template <GameLike G>
GameRecord<G> play_selfplay_game(const SelfPlayConfig& cfg, Evaluator<G>& ev, Rng& rng) {
  GameRecord<G> rec; MCTS<G> mcts(cfg.mcts); auto s = G::initial(); mcts.set_root(s);
  while (!G::is_terminal(s) && rec.plies < G::kMaxGameLength) {
    Move mv[G::kMaxMoves];
    int nm = G::legal_moves(s, mv);
    // Tier-A cheap shortcuts: no search, value-only targets.
    if (cfg.enable_shortcuts && nm == 1) {
      G::apply(s, mv[0]); rec.plies++;
      mcts.set_root(s);
      continue;  // forced move teaches nothing about preference: emit no sample
    }
    Move sc = -1;
    if (cfg.enable_shortcuts && (find_win_in_1<G>(s, &sc) || find_block_in_1<G>(s, &sc))) {
      Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount, 0.f);
      sm.player = G::current_player(s); sm.full_search = 0.f;
      G::encode(s, sm.input.data());
      if (sc >= 0 && sc < G::kActionCount) sm.policy[sc] = 1.f;
      rec.samples.push_back(std::move(sm));
      G::apply(s, sc); rec.plies++;
      mcts.set_root(s);
      continue;
    }
    bool full = rng.uniform() < cfg.full_search_prob;
    int sims = cfg.simulations;
    if (!full) {
      sims = std::max(cfg.cheap_min_sims, (int)(cfg.simulations * cfg.cheap_sim_fraction));
    }
    int have = mcts.root_expanded() ? mcts.total_visits() : 0;
    if (!mcts.root_expanded()) mcts.run(1, ev, 1);
    mcts.add_root_noise(rng);
    mcts.run(std::max(0, sims - have), ev, cfg.batch);
    // Proven root win: stop searching this position (already solved).
    Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount); sm.player = G::current_player(s);
    sm.full_search = full ? 1.f : 0.f;
    G::encode(s, sm.input.data());
    if (cfg.policy_pruning) mcts.improved_policy(sm.policy.data());
    else mcts.visit_policy(sm.policy.data());
    Move m = mcts.sample_move(rec.plies < cfg.temperature_moves ? cfg.temperature : 0.f, rng);
    rec.samples.push_back(std::move(sm));
    G::apply(s, m); rec.plies++;
    if (cfg.reuse_tree) mcts.advance_root(m); else mcts.set_root(s);
  }
  rec.outcome_p0 = G::outcome(s, 0);
  for (auto& sm : rec.samples) sm.value = G::outcome(s, sm.player);
  return rec;
}

template <GameLike G> class SampleWriter {
 public:
  explicit SampleWriter(const char* path) {
    f_ = std::fopen(path, "wb"); if (!f_) return;
    uint32_t h[5] = {3u, (uint32_t)G::kInputPlanes, (uint32_t)G::kInputH, (uint32_t)G::kInputW, (uint32_t)G::kActionCount};
    std::fwrite("GAIS", 1, 4, f_); std::fwrite(h, 4, 5, f_);
  }
  ~SampleWriter() { if (f_) std::fclose(f_); }
  bool ok() const { return f_ != nullptr; }
  void write(const GameRecord<G>& r) {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& s : r.samples) { std::fwrite(s.input.data(), 4, s.input.size(), f_); std::fwrite(s.policy.data(), 4, s.policy.size(), f_); std::fwrite(&s.value, 4, 1, f_); std::fwrite(&s.full_search, 4, 1, f_); }
    positions_ += r.samples.size();
  }
  size_t positions() const { return positions_; }
 private:
  FILE* f_ = nullptr; std::mutex mu_; size_t positions_ = 0;
};
}  // namespace gai
