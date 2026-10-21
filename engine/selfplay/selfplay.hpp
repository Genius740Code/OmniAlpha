// Self-play game generation + binary sample file format (read by python/gai/data.py).
// File: magic "GAIS", then u32 x5: version(4), planes, height, width, action_count; then records of
//   float input[planes*h*w]  (encode() of the state, view of player to move)
//   float policy[action_count] (MCTS visit distribution; one-hot for tactical shortcuts)
//   float value               (final outcome from the view of the player to move, +1/0/-1)
//   float full                (1 = full-budget search, policy target valid; 0 = cheap/shortcut, value-only)
//   float q                   (search root_value at move time, player-to-move view; for outcome mixing)
#pragma once
#include <cstdio>
#include <mutex>
#include <vector>
#include "engine/mcts/gumbel_root.hpp"
#include "engine/mcts/mcts.hpp"

namespace gai {
struct SelfPlayConfig {
  int simulations = 400;
  int batch = 1;
  int temperature_moves = 10;   // sample ~visits^(1/T) for the first N plies, then argmax
  float temperature = 1.0f;
  bool reuse_tree = true;
  bool enable_shortcuts = true;   // single-move skip + win-in-1 / block-in-1
  bool tactics2 = false;          // opt-in 2-ply forced-win (win-in-3) root shortcut; see
                                  // find_forced_win_2ply below
  bool policy_pruning = true;     // KataGo-style: train on pruned target, play on raw visits
  float full_search_prob = 1.0f;  // playout-cap randomization: P(full sims); else cheap
  float cheap_sim_fraction = 0.125f;
  int cheap_min_sims = 10;
  float value_lambda = 0.1f;      // soft-Z: blend outcome with search Q [0,1]
                                  // (matches tools/selfplay --value-lambda and orchestrator defaults)
  float resign_q = -2.0f;         // resign when best child Q < this; <=-1.5 disables
  int resign_min_plies = 10;      // don't resign before this many plies
  float resign_probe = 0.0f;      // fraction of hopeless positions to play on (calibration: measures false-resign rate)
  float truncate_value = 0.0f;    // value (p0 view) for games that stop at kMaxGameLength without
                                  // being terminal (adjudicated draw by default). G::outcome() on a
                                  // non-terminal state is meaningless, so truncation must not use it.
                                  // Connect4 fills the board by ply 42, so this only bites long games.
  int lanes = 1;                  // lockstep games per worker (>1 batches one leaf
                                  // per game into a single evaluator call; no virtual
                                  // loss, unchanged per-tree semantics; non-Gumbel only)
  MctsConfig mcts;
};

template <GameLike G> struct Sample {
  std::vector<float> input, policy; int player; float value = 0; float full_search = 1.f;
  float q = 0.f;  // search value (root_value) at move time, player-to-move view; for mixing
};
template <GameLike G> struct GameRecord { std::vector<Sample<G>> samples; float outcome_p0 = 0; int plies = 0;
  int tactic2_solves = 0;  // positions solved by the 2-ply forced-win shortcut (reporting only)
};

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
    if (G::is_terminal(t)) { continue; }  // own terminal here is non-winning (win-in-1 already false): not a safe block
    if (!find_win_in_1<G>(t, &win)) { safe++; last_safe = mv[i]; }
  }
  if (safe == 1) { *out = last_safe; return true; }
  return false;
}

// 2-ply forced win ("win-in-3"): a root move m such that EVERY opponent reply to m
// leaves us an immediate win-in-1 (or loses/draws on the spot is rejected -- only
// replies that lose immediately or still lose to our win-in-1 pass). Proven win,
// no search needed: cost is root_moves * replies * replies win-in-1 scans
// (Connect4 ~ 7*7*7 ~ 350 cheap apply/legality checks per call).
template <GameLike G> bool find_forced_win_2ply(const typename G::State& s, Move* out) {
  Move mv[G::kMaxMoves];
  int n = G::legal_moves(s, mv);
  int me = G::current_player(s);
  for (int i = 0; i < n; i++) {
    auto t = s;
    G::apply(t, mv[i]);
    if (G::is_terminal(t)) {
      // Ends the game now: a win here is win-in-1's job (kept for standalone use);
      // a draw/loss is not a forced win.
      if (G::outcome(t, me) > 0) { *out = mv[i]; return true; }
      continue;
    }
    Move rm[G::kMaxMoves];
    int rn = G::legal_moves(t, rm);
    bool forced = rn > 0;
    for (int j = 0; j < rn && forced; j++) {
      auto u = t;
      G::apply(u, rm[j]);
      if (G::is_terminal(u)) {
        // Opponent reply that loses on the spot is fine; a draw or an
        // opponent win means m is refuted.
        if (G::outcome(u, me) <= 0) forced = false;
        continue;
      }
      Move w;
      if (!find_win_in_1<G>(u, &w)) forced = false;  // opponent survives our next move
    }
    if (forced) { *out = mv[i]; return true; }
  }
  return false;
}

template <GameLike G>
GameRecord<G> play_selfplay_game(const SelfPlayConfig& cfg, Evaluator<G>& ev, Rng& rng) {
  GameRecord<G> rec; MCTS<G> mcts(cfg.mcts); auto s = G::initial(); mcts.set_root(s);
  bool resigned = false; int resign_loser = -1;
  while (!G::is_terminal(s) && rec.plies < G::kMaxGameLength) {
    Move mv[G::kMaxMoves];
    int nm = G::legal_moves(s, mv);
    // Tier-A cheap shortcuts: no search, value-only targets.
    if (cfg.enable_shortcuts && nm == 1) {
      G::apply(s, mv[0]); rec.plies++;
      mcts.set_root(s);
      continue;  // forced move teaches nothing about preference: emit no sample
    }
    Move sc = -1; bool is_win = false;
    if (cfg.enable_shortcuts && ((is_win = find_win_in_1<G>(s, &sc)) || find_block_in_1<G>(s, &sc))) {
      Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount, 0.f);
      sm.player = G::current_player(s); sm.full_search = 0.f;
      sm.q = is_win ? 1.f : 0.f;  // win-in-1 is certain; block is neutral prior
      G::encode(s, sm.input.data());
      if (sc >= 0 && sc < G::kActionCount) sm.policy[sc] = 1.f;
      rec.samples.push_back(std::move(sm));
      G::apply(s, sc); rec.plies++;
      mcts.set_root(s);
      continue;
    }
    // Opt-in 2-ply tactics: proven win-in-3 at the root. Same sample treatment as
    // the win-in-1 shortcut (one-hot policy target, value-only, certain q); the
    // search is skipped entirely.
    if (cfg.tactics2) {
      Move fw = -1;
      if (find_forced_win_2ply<G>(s, &fw)) {
        Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount, 0.f);
        sm.player = G::current_player(s); sm.full_search = 0.f;
        sm.q = 1.f;  // forced win is certain
        G::encode(s, sm.input.data());
        if (fw >= 0 && fw < G::kActionCount) sm.policy[fw] = 1.f;
        rec.tactic2_solves++;
        rec.samples.push_back(std::move(sm));
        G::apply(s, fw); rec.plies++;
        mcts.set_root(s);
        continue;
      }
    }
    bool full = rng.uniform() < cfg.full_search_prob;
    int sims = cfg.simulations;
    if (!full) {
      sims = std::max(cfg.cheap_min_sims, (int)(cfg.simulations * cfg.cheap_sim_fraction));
    }
    int have = mcts.root_expanded() ? mcts.total_visits() : 0;
    if (!mcts.root_expanded()) mcts.run(1, ev, 1);
    Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount); sm.player = G::current_player(s);
    sm.full_search = full ? 1.f : 0.f;
    G::encode(s, sm.input.data());
    Move m = -1;
    if (cfg.mcts.gumbel && full) {
      // Real Gumbel root: replaces Dirichlet noise, forced playouts and
      // temperature exploration. Target is the improved policy (guaranteed
      // improvement even at few sims); every move uses the small budget.
      auto gr = run_gumbel_root(mcts, ev, cfg.mcts, rng, cfg.batch);
      sm.policy = gr.improved;
      float temp = rec.plies < cfg.temperature_moves ? cfg.temperature : 0.f;
      if (temp <= 1e-3f || gr.best < 0) m = gr.best;
      if (m < 0) {
        Move mv[G::kMaxMoves];
        int n = G::legal_moves(s, mv);
        double sum = 0;
        for (int i = 0; i < n; i++) sum += std::pow((double)std::max(sm.policy[mv[i]], 0.f), 1.0 / temp);
        double r = rng.uniform() * sum;
        m = mv[n - 1];
        for (int i = 0; i < n; i++) {
          r -= std::pow((double)std::max(sm.policy[mv[i]], 0.f), 1.0 / temp);
          if (r <= 0) { m = mv[i]; break; }
        }
      }
    } else {
      mcts.add_root_noise(rng);
      mcts.run(std::max(0, sims - have), ev, cfg.batch);
      // Proven root win: stop searching this position (already solved).
      if (cfg.policy_pruning) mcts.improved_policy(sm.policy.data());
      else mcts.visit_policy(sm.policy.data());
      m = mcts.sample_move(rec.plies < cfg.temperature_moves ? cfg.temperature : 0.f, rng);
    }
    sm.q = mcts.root_value();  // search root-value at move time, for value-target mixing
    // Calibrated resignation: hopeless full-search positions end now (probe fraction plays on
    // to measure the false-resign rate); loser = player to move.
    if (!resigned && cfg.resign_q > -1.5f && full && rec.plies >= cfg.resign_min_plies
        && mcts.best_child_q() < cfg.resign_q && rng.uniform() >= cfg.resign_probe) {
      resigned = true; resign_loser = sm.player;
      rec.samples.push_back(std::move(sm));
      break;
    }
    rec.samples.push_back(std::move(sm));
    G::apply(s, m); rec.plies++;
    if (cfg.reuse_tree) mcts.advance_root(m); else mcts.set_root(s);
  }
  if (resigned) {
    rec.outcome_p0 = (resign_loser == 0 ? -1.f : 1.f);
  } else if (!G::is_terminal(s)) {
    // Truncated at kMaxGameLength: outcome() of a non-terminal state is
    // meaningless; use the adjudicated value instead (audit A2).
    rec.outcome_p0 = cfg.truncate_value;
  } else {
    rec.outcome_p0 = G::outcome(s, 0);
  }
  for (auto& sm : rec.samples) {
    // soft-Z style: blend game outcome with search root-value Q.
    // Both terms are player-to-move view (outcome(s, sm.player); q = root_value
    // from the chooser's view). NOTE: outcome(s, 0) is p0-view and would flip
    // the sign for sm.player == 1 samples -- do NOT use it here.
    // truncate_value is p0-view, so flip the sign for player-1 samples.
    float z;
    if (resigned) z = (sm.player == resign_loser ? -1.f : 1.f);
    else if (!G::is_terminal(s)) z = (sm.player == 0 ? cfg.truncate_value : -cfg.truncate_value);
    else z = G::outcome(s, sm.player);
    sm.value = (1.f - cfg.value_lambda) * z + cfg.value_lambda * sm.q;
  }
  return rec;
}

// Lane-batched selfplay: L games stepped in lockstep. Per-move logic mirrors
// play_selfplay_game exactly (shortcuts, full/cheap, noise, temperature, resign,
// value mixing); only the search phase differs: one leaf per unfinished game is
// selected per round and all leaves share a single evaluator call. No virtual
// loss, unchanged per-tree semantics. Non-Gumbel path only: Gumbel uses its own
// internal batching (run_gumbel_root), so Gumbel+lanes falls back to per-game.
// Returns one GameRecord per lane (all finalized with blended values).
template <GameLike G>
std::vector<GameRecord<G>> play_selfplay_lanes(const SelfPlayConfig& cfg, Evaluator<G>& ev, Rng& rng, int L) {
  struct Slot {
    MCTS<G> m; typename G::State s; GameRecord<G> rec;
    bool active = true, resigned = false, truncated = false; int resign_loser = -1;
    bool has_sm = false, full = false; int sims_left = 0; Sample<G> sm;
  };
  std::vector<Slot> sl(L);
  for (auto& t : sl) { t.m = MCTS<G>(cfg.mcts); t.s = G::initial(); t.m.set_root(t.s); }
  auto finalize = [&](Slot& t) {
    if (t.resigned) t.rec.outcome_p0 = (t.resign_loser == 0 ? -1.f : 1.f);
    else if (t.truncated) t.rec.outcome_p0 = cfg.truncate_value;  // non-terminal: adjudicated (audit A2)
    else t.rec.outcome_p0 = G::outcome(t.s, 0);
    for (auto& sm : t.rec.samples) {
      // truncate_value is p0-view, so flip the sign for player-1 samples.
      float z;
      if (t.resigned) z = (sm.player == t.resign_loser ? -1.f : 1.f);
      else if (t.truncated) z = (sm.player == 0 ? cfg.truncate_value : -cfg.truncate_value);
      else z = G::outcome(t.s, sm.player);
      sm.value = (1.f - cfg.value_lambda) * z + cfg.value_lambda * sm.q;
    }
    t.active = false;
  };
  int alive = L;
  Move mv[G::kMaxMoves];
  while (alive > 0) {
    // Phase A: per-slot move prelude (identical logic to play_selfplay_game).
    for (auto& t : sl) {
      if (!t.active || t.has_sm) continue;
      if (G::is_terminal(t.s) || t.rec.plies >= G::kMaxGameLength) {
        t.truncated = !G::is_terminal(t.s);
        finalize(t); alive--; continue;
      }
      int nm = G::legal_moves(t.s, mv);
      if (cfg.enable_shortcuts && nm == 1) {
        G::apply(t.s, mv[0]); t.rec.plies++;
        t.m.set_root(t.s);
        continue;  // no sample; slot needs another prelude next pass
      }
      Move sc = -1; bool is_win = false;
      if (cfg.enable_shortcuts && ((is_win = find_win_in_1<G>(t.s, &sc)) || find_block_in_1<G>(t.s, &sc))) {
        Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount, 0.f);
        sm.player = G::current_player(t.s); sm.full_search = 0.f;
        sm.q = is_win ? 1.f : 0.f;
        G::encode(t.s, sm.input.data());
        if (sc >= 0 && sc < G::kActionCount) sm.policy[sc] = 1.f;
        t.rec.samples.push_back(std::move(sm));
        G::apply(t.s, sc); t.rec.plies++;
        t.m.set_root(t.s);
        continue;
      }
      // Opt-in 2-ply tactics: identical treatment to play_selfplay_game above.
      if (cfg.tactics2) {
        Move fw = -1;
        if (find_forced_win_2ply<G>(t.s, &fw)) {
          Sample<G> sm; sm.input.resize(input_size<G>()); sm.policy.resize(G::kActionCount, 0.f);
          sm.player = G::current_player(t.s); sm.full_search = 0.f;
          sm.q = 1.f;
          G::encode(t.s, sm.input.data());
          if (fw >= 0 && fw < G::kActionCount) sm.policy[fw] = 1.f;
          t.rec.tactic2_solves++;
          t.rec.samples.push_back(std::move(sm));
          G::apply(t.s, fw); t.rec.plies++;
          t.m.set_root(t.s);
          continue;
        }
      }
      t.full = rng.uniform() < cfg.full_search_prob;
      int sims = cfg.simulations;
      if (!t.full) sims = std::max(cfg.cheap_min_sims, (int)(cfg.simulations * cfg.cheap_sim_fraction));
      int have = t.m.root_expanded() ? t.m.total_visits() : 0;
      if (!t.m.root_expanded()) t.m.run(1, ev, 1);
      t.sm = Sample<G>(); t.sm.input.resize(input_size<G>()); t.sm.policy.resize(G::kActionCount);
      t.sm.player = G::current_player(t.s);
      t.sm.full_search = t.full ? 1.f : 0.f;
      G::encode(t.s, t.sm.input.data());
      t.m.add_root_noise(rng);
      t.sims_left = std::max(0, sims - have);
      t.has_sm = true;
    }
    // Phase B: lockstep batched search, one leaf per unfinished slot per round.
    bool need = true;
    while (need) {
      need = false;
      std::vector<std::pair<int, typename MCTS<G>::Leaf>> picks;
      std::vector<typename G::State> sts;
      for (int i = 0; i < L; i++) {
        if (!sl[i].active || !sl[i].has_sm || sl[i].sims_left <= 0) continue;
        need = true;
        auto lf = sl[i].m.select_leaf();
        if (!lf.terminal) sts.push_back(lf.state);
        picks.emplace_back(i, lf);
      }
      if (picks.empty()) break;
      std::vector<EvalResult<G>> res(sts.size());
      if (!sts.empty()) ev.evaluate(sts.data(), res.data(), (int)sts.size());
      size_t k = 0;
      for (auto& [i, lf] : picks) {
        sl[i].m.finish_leaf(lf, lf.terminal ? nullptr : &res[k++]);
        sl[i].sims_left--;
      }
    }
    // Phase C: per-slot move selection (identical logic to play_selfplay_game).
    for (auto& t : sl) {
      if (!t.active || !t.has_sm) continue;
      t.has_sm = false;
      if (cfg.policy_pruning) t.m.improved_policy(t.sm.policy.data());
      else t.m.visit_policy(t.sm.policy.data());
      Move m = t.m.sample_move(t.rec.plies < cfg.temperature_moves ? cfg.temperature : 0.f, rng);
      t.sm.q = t.m.root_value();
      if (!t.resigned && cfg.resign_q > -1.5f && t.full && t.rec.plies >= cfg.resign_min_plies
          && t.m.best_child_q() < cfg.resign_q && rng.uniform() >= cfg.resign_probe) {
        t.resigned = true; t.resign_loser = t.sm.player;
        t.rec.samples.push_back(std::move(t.sm));
        finalize(t); alive--;
        continue;
      }
      t.rec.samples.push_back(std::move(t.sm));
      G::apply(t.s, m); t.rec.plies++;
      if (cfg.reuse_tree) t.m.advance_root(m); else t.m.set_root(t.s);
    }
  }
  std::vector<GameRecord<G>> out(L);
  for (int i = 0; i < L; i++) out[i] = std::move(sl[i].rec);
  return out;
}

template <GameLike G> class SampleWriter {
 public:
  explicit SampleWriter(const char* path) {
    f_ = std::fopen(path, "wb"); if (!f_) return;
    uint32_t h[5] = {4u, (uint32_t)G::kInputPlanes, (uint32_t)G::kInputH, (uint32_t)G::kInputW, (uint32_t)G::kActionCount};
    std::fwrite("GAIS", 1, 4, f_); std::fwrite(h, 4, 5, f_);
  }
  ~SampleWriter() { if (f_) std::fclose(f_); }
  bool ok() const { return f_ != nullptr; }
  void write(const GameRecord<G>& r) {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& s : r.samples) { std::fwrite(s.input.data(), 4, s.input.size(), f_); std::fwrite(s.policy.data(), 4, s.policy.size(), f_); std::fwrite(&s.value, 4, 1, f_); std::fwrite(&s.full_search, 4, 1, f_); std::fwrite(&s.q, 4, 1, f_); }
    positions_ += r.samples.size();
  }
  size_t positions() const { return positions_; }
 private:
  FILE* f_ = nullptr; std::mutex mu_; size_t positions_ = 0;
};
}  // namespace gai
