// Naive Connect-4: cell array + full-board scan. Deliberately different from the bitboard code.
#pragma once
#include <array>
#include "engine/core/api.hpp"
namespace game_connect4 {
struct RefState { std::array<int8_t, 42> cell{}; int plies = 0; };  // cell[r*7+c], r=0 bottom; 0 empty, 1 = P0, 2 = P1
struct RefGame {
  using State = RefState;
  static State initial() { return {}; }
  static int legal_moves(const State& s, gai::Move* out) { int k = 0; for (int c = 0; c < 7; c++) if (s.cell[5 * 7 + c] == 0) out[k++] = (gai::Move)c; return k; }
  static void apply(State& s, gai::Move c) { for (int r = 0; r < 6; r++) if (s.cell[r * 7 + c] == 0) { s.cell[r * 7 + c] = (int8_t)(1 + (s.plies % 2)); break; } s.plies++; }
  static int winner(const State& s) {  // 0 none, else 1/2
    const int dr[4] = {0, 1, 1, 1}, dc[4] = {1, 0, 1, -1};
    for (int r = 0; r < 6; r++) for (int c = 0; c < 7; c++) { int v = s.cell[r * 7 + c]; if (!v) continue;
      for (int d = 0; d < 4; d++) { int k = 1; for (; k < 4; k++) { int rr = r + dr[d] * k, cc = c + dc[d] * k; if (rr < 0 || rr >= 6 || cc < 0 || cc >= 7 || s.cell[rr * 7 + cc] != v) break; } if (k == 4) return v; } }
    return 0;
  }
  static bool is_terminal(const State& s) { return winner(s) != 0 || s.plies == 42; }
  static int current_player(const State& s) { return s.plies % 2; }
  static float outcome(const State& s, int p) { int w = winner(s); if (!w) return 0.f; return (w - 1) == p ? 1.f : -1.f; }
};
}  // namespace game_connect4
