// Slow, obviously-correct board implementation for differential testing against Game.
// Written with plain char grid + full-board rescan ON PURPOSE (shared-bug insurance).
#pragma once
#include <cstring>
#include <vector>
#include "engine/core/api.hpp"
namespace game_demo_board {
struct RefState {
  char cell[N];
  int n = 0;
  RefState() { std::memset(cell, '.', sizeof(cell)); }
};
struct RefGame {
  using State = RefState;
  static char at(const State& s, int x, int y) { return s.cell[y * W + x]; }
  static char char_for_player(int p) { return p == 0 ? 'X' : 'O'; }
  static char mover_char(const State& s) { return char_for_player(s.n % 2); }       // the player to move
  static char last_mover_char(const State& s) { return char_for_player((s.n - 1) % 2); }
  static State initial() { return {}; }
  static int legal_moves(const State& s, gai::Move* out) {
    int k = 0;
    for (int i = 0; i < N; i++) if (s.cell[i] == '.') out[k++] = (gai::Move)i;
    return k;
  }
  static void apply(State& s, gai::Move c) { s.cell[c] = mover_char(s); s.n++; }
  static bool is_terminal(const State& s) {
    if (s.n == N) return true;
    const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    char who = last_mover_char(s);  // mover of the LAST move
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      if (at(s, x, y) != who) continue;
      for (auto& d : dirs) {
        int run = 0, ok = true;
        for (int i = 0; i < K; i++) { int xx = x + d[0] * i, yy = y + d[1] * i;
          if (xx >= W || yy >= H || xx < 0 || yy < 0 || at(s, xx, yy) != who) { ok = false; break; } run++; }
        if (ok && run >= K) return true;
      }
    }
    return false;
  }
  static int current_player(const State& s) { return s.n % 2; }
  static float outcome(const State& s, int p) {
    if (!is_terminal(s)) return 0.f;
    // winner = whoever moved last, but only if they actually made a line (else a full-board draw)
    char last = last_mover_char(s);
    const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    bool won = false;
    for (int y = 0; y < H && !won; y++) for (int x = 0; x < W && !won; x++) {
      if (at(s, x, y) != last) continue;
      for (auto& d : dirs) {
        bool ok = true;
        for (int i = 0; i < K; i++) { int xx = x + d[0] * i, yy = y + d[1] * i;
          if (xx >= W || yy >= H || xx < 0 || yy < 0 || at(s, xx, yy) != last) { ok = false; break; } }
        won |= ok;
      }
    }
    if (!won) return 0.f;
    int winner = (s.n - 1) % 2;
    return p == winner ? 1.f : -1.f;
  }
};
}  // namespace game_demo_board
