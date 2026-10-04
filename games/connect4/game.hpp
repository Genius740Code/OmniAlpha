// Connect-4 (7x6), optimized bitboard implementation (Tromp layout: 7 bits per column, bit 6 is a sentinel).
// Move = column 0..6. Policy index == column.
#pragma once
#include <bit>
#include <string>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"

namespace game_connect4 {
using gai::Move;
constexpr int W = 7, H = 6;

struct State {
  uint64_t cur = 0;   // stones of the player TO MOVE
  uint64_t mask = 0;  // all stones
  uint8_t n = 0;      // plies played
  uint8_t won = 0;    // 1 if the player who just moved completed 4-in-row
};

struct Game {
  using State = ::game_connect4::State;
  static constexpr int kActionCount = W, kMaxMoves = W, kMaxGameLength = W * H;
  static constexpr int kInputPlanes = 2, kInputH = H, kInputW = W;
  static constexpr int kNumSymmetries = 2;  // identity + horizontal mirror

  static constexpr uint64_t bottom(int c) { return 1ULL << (c * 7); }
  static constexpr uint64_t topbit(int c) { return 1ULL << (c * 7 + H - 1); }
  static bool align4(uint64_t p) {
    uint64_t m = p & (p >> 7); if (m & (m >> 14)) return true;  // horizontal
    m = p & (p >> 6);          if (m & (m >> 12)) return true;  // diagonal
    m = p & (p >> 8);          if (m & (m >> 16)) return true;  // anti-diagonal
    m = p & (p >> 1);          if (m & (m >> 2)) return true;   // vertical
    return false;
  }

  static State initial() { return State{}; }
  static int legal_moves(const State& s, Move* out) {
    int k = 0; for (int c = 0; c < W; c++) if (!(s.mask & topbit(c))) out[k++] = (Move)c; return k;
  }
  static void apply(State& s, Move c) {
    uint64_t nm = s.mask | (s.mask + bottom(c));
    uint64_t mover = s.cur | (nm & ~s.mask);
    s.won = align4(mover) ? 1 : 0;
    s.cur ^= s.mask; s.mask = nm; s.n++;
  }
  static void undo(State& s, Move c) {
    uint64_t col = s.mask & (((1ULL << H) - 1) << (c * 7));
    uint64_t top = 1ULL << (63 - std::countl_zero(col));
    s.mask ^= top; s.cur ^= s.mask; s.n--; s.won = 0;
  }
  static bool is_terminal(const State& s) { return s.won || s.n == W * H; }
  static int current_player(const State& s) { return s.n & 1; }
  static float outcome(const State& s, int p) {
    if (!s.won) return 0.f;
    int winner = (s.n - 1) & 1; return p == winner ? 1.f : -1.f;
  }
  static gai::Hash hash(const State& s) { return gai::mix64(s.cur ^ gai::mix64(s.mask)); }
  // plane 0: stones of player to move, plane 1: opponent stones. index [plane][row(0=bottom)][col].
  static void encode(const State& s, float* out) {
    uint64_t mine = s.cur, theirs = s.cur ^ s.mask;
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
      uint64_t b = 1ULL << (c * 7 + r);
      out[r * W + c] = (mine & b) ? 1.f : 0.f;
      out[H * W + r * W + c] = (theirs & b) ? 1.f : 0.f;
    }
  }
  static Move transform_move(Move m, int sym) { return sym == 0 ? m : (Move)(W - 1 - m); }
  static void symmetry_input(const float* in, float* out, int sym) {
    for (int p = 0; p < kInputPlanes; p++) for (int r = 0; r < H; r++) for (int c = 0; c < W; c++)
      out[(p * H + r) * W + (sym == 0 ? c : W - 1 - c)] = in[(p * H + r) * W + c];
  }
  static std::string to_string(const State& s) {
    std::string t; uint64_t p0 = (s.n & 1) ? (s.cur ^ s.mask) : s.cur;  // player 0 stones
    for (int r = H - 1; r >= 0; r--) { for (int c = 0; c < W; c++) { uint64_t b = 1ULL << (c * 7 + r); t += (s.mask & b) ? ((p0 & b) ? 'X' : 'O') : '.'; } t += '\n'; }
    t += "0123456\n"; return t;
  }
};
}  // namespace game_connect4
