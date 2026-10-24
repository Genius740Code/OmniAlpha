// TEMPLATE: chance + capture RACE game ("dice" family: monopoly/ludo/snakes-like).
// Scaffold with: python3 tools/new_game.py <name> --from dice
//
// HOW CHANCE WORKS WITHOUT ENGINE CHANGES: dice are a deterministic hash-chain
// (`roll = f(chain)`), updated every move. Positions + chain fully determine the
// game, so MCTS/validator/NN stay correct; but because the chain also encodes the
// move history, rolls diverge between games (the chain depends on what was played).
// Cost: the first roll of every game is identical -> games share an opening like
// chess does; root Dirichlet noise keeps self-play exploration sane (already on).
//
// Rules: path of 30 steps, exact landing to win. Each turn: roll r in 1..6 (from
// the chain), mover takes k in 1..r steps (move id == k; id 0 = "pass", legal ONLY
// when every step would overshoot 30). Landing on 7/14/21/28 slides +2; landing on
// the opponent sends them back 3.
//
// TODO: replace squares/bonuses/captures with YOUR board, keep: dice = f(chain),
// move id == steps, pass only when stuck. See docs/ADD_ANY_GAME.md for the recipe
// table (e.g. "monopoly cards" = another hash-chain; "may choose dice" is NOT fair).
#pragma once
#include <cstring>
#include <string>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"

namespace game_template {
using gai::Move;
constexpr int kGoal = 30;  // TODO: your track length

struct State {
  uint64_t chain = 0x9E3779B97F4A7C15ULL;  // public dice deck: roll = f(chain)
  uint8_t pos[2] = {0, 0};                 // 'pos' from the start
  uint8_t player = 0;
};

struct Game {
  using State = ::game_template::State;
  static constexpr int kActionCount = 7;   // policy index == step count 0(pass)..6
  static constexpr int kMaxMoves = 7;
  static constexpr int kMaxGameLength = 400;
  static constexpr int kInputPlanes = 3, kInputH = 1, kInputW = kGoal + 1;  // 2 pos + roll
  static constexpr int kNumSymmetries = 1;  // races are asymmetric

  static State initial() { return State{}; }

  static int roll(const State& s) {
    uint64_t h = gai::mix64(s.chain * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL);
    return 1 + (int)((h >> 25) % 6);
  }

  static int legal_moves(const State& s, Move* out) {
    int r = roll(s), me = s.player, n = 0;
    for (int k = 1; k <= r; k++) if ((int)s.pos[me] + k <= kGoal) out[n++] = (Move)k;
    if (n == 0) out[n++] = 0;  // pass ONLY when every step would overshoot
    return n;
  }

  static void apply(State& s, Move m) {
    int me = s.player, op = me ^ 1, k = m;
    if (k > 0) {  // 0 = pass: still advances the dice chain below
      int p = s.pos[me] + k;
      if (p > 0 && p < kGoal && p % 7 == 0) p += 2;          // bonus square slide
      if (p > kGoal) p = kGoal;                              // slide can reach/exceed goal
      s.pos[me] = (uint8_t)p;
      if (s.pos[op] == s.pos[me] && s.pos[me] > 0)           // capture: send back 3
        s.pos[op] = (uint8_t)((s.pos[op] > 3) ? s.pos[op] - 3 : 0);
    }
    s.chain = gai::mix64(s.chain + 0x2545F4914F6CDD1DULL * (uint64_t)(k + 1));
    s.player ^= 1;
  }

  static bool is_terminal(const State& s) { return s.pos[0] == kGoal || s.pos[1] == kGoal; }
  static int current_player(const State& s) { return s.player; }
  static float outcome(const State& s, int p) {
    int winner = s.pos[0] == kGoal ? 0 : 1;
    return p == winner ? 1.f : -1.f;
  }
  static gai::Hash hash(const State& s) {
    return gai::mix64(s.chain ^ ((uint64_t)s.pos[0] << 32) ^ ((uint64_t)s.pos[1] << 40) ^ ((uint64_t)s.player << 48));
  }

  // plane 0 = my position (one-hot), plane 1 = opponent, plane 2 = current roll (one-hot at index r).
  static void encode(const State& s, float* out) {
    std::memset(out, 0, sizeof(float) * 3 * kInputW);
    int p = s.player;
    out[s.pos[p]] = 1.f;                                   // careful: pos may hit goal? (terminal, never encoded)
    out[kInputW + s.pos[p ^ 1]] = 1.f;
    out[2 * kInputW + roll(s)] = 1.f;
  }

  // Required by the API even with kNumSymmetries == 1: identity hooks.
  static Move transform_move(Move m, int) { return m; }
  static void symmetry_input(const float* in, float* out, int) { std::memcpy(out, in, sizeof(float) * 3 * kInputW); }

  static std::string to_string(const State& s) {
    std::string t = "P0[";
    for (int i = 0; i <= kGoal; i++) t += (i == s.pos[0] ? "0" : (i % 7 == 0 && i > 0 && i < kGoal ? "*" : (i == kGoal ? "|" : "-")));
    t += "]\nP1[";
    for (int i = 0; i <= kGoal; i++) t += (i == s.pos[1] ? "1" : (i % 7 == 0 && i > 0 && i < kGoal ? "*" : (i == kGoal ? "|" : "-")));
    t += "] roll=" + std::to_string(roll(s)) + " player " + std::to_string(s.player) + " (move id = steps 1.." +
         std::to_string(roll(s)) + ", 0 only if stuck)\n";
    return t;
  }
};
}  // namespace game_template
