// TEMPLATE GAME: "Take-away". Pile of N stones, players alternate taking 1..3, whoever takes the last stone wins.
// Copy this directory to games/<your_game>/, rename the namespace to game_<your_game>, and replace the logic.
// Read docs/ADDING_A_GAME.md. Required: game.hpp (this file). Strongly recommended: reference_game.hpp.
#pragma once
#include <cstring>
#include <string>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"

namespace game_template {
using gai::Move;

struct State { uint8_t pile = 21; uint8_t player = 0; };  // keep State tiny & trivially copyable

struct Game {
  using State = ::game_template::State;
  // ---- constants (TODO: set for your game) ----
  static constexpr int kActionCount = 3;      // policy size; Move value == policy index
  static constexpr int kMaxMoves = 3;         // max legal moves in any position
  static constexpr int kMaxGameLength = 21;   // hard upper bound on plies (validator flags longer games)
  static constexpr int kInputPlanes = 1, kInputH = 1, kInputW = 22;  // NN input layout [plane][h][w]
  static constexpr int kNumSymmetries = 1;    // >1 only if you implement the two symmetry hooks below

  static State initial() { return State{}; }
  static int legal_moves(const State& s, Move* out) {  // write moves in ascending order, return count
    int n = 0; for (int k = 1; k <= 3 && k <= s.pile; k++) out[n++] = (Move)(k - 1); return n;
  }
  static void apply(State& s, Move m) { s.pile -= (uint8_t)(m + 1); s.player ^= 1; }
  static bool is_terminal(const State& s) { return s.pile == 0; }
  static int current_player(const State& s) { return s.player; }
  // +1 win / 0 draw / -1 loss for `p`. Only meaningful in terminal states. Must be zero-sum.
  static float outcome(const State& s, int p) {
    if (s.pile != 0) return 0.f;
    int winner = s.player ^ 1;  // last mover took the last stone
    return p == winner ? 1.f : -1.f;
  }
  static gai::Hash hash(const State& s) { return gai::mix64(((uint64_t)s.pile << 1) | s.player); }
  // Encode from the view of the player to move. Layout [plane][h][w]; zero what you don't set.
  static void encode(const State& s, float* out) {
    std::memset(out, 0, sizeof(float) * 22); out[s.pile] = 1.f;
  }
  // ---- optional symmetries (only used if kNumSymmetries > 1) ----
  static Move transform_move(Move m, int /*sym*/) { return m; }
  static void symmetry_input(const float* in, float* out, int /*sym*/) { std::memcpy(out, in, sizeof(float) * 22); }
  static std::string to_string(const State& s) { return "pile=" + std::to_string(s.pile) + " to_move=" + std::to_string(s.player); }
};
}  // namespace game_template
