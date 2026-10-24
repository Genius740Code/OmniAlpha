// TEMPLATE: square-grid win-in-a-row board game (bitboard, 8x8, connect-K=4).
// Copy with: python3 tools/new_game.py <name> --from board
// Then adjust the 6 constants below and the to_string/encode row order documentation.
// This family also covers: othello (add flip-set updates to apply), chess-lite
// (encode piece planes + caret moves), gomoku (bigger board + bigger K).
#pragma once
#include <cstring>
#include <string>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"

namespace game_demo_board {
using gai::Move;
constexpr int W = 8, H = 8, K = 4;             // TODO board dims + win length
constexpr int N = W * H;

struct State {
  uint64_t cur = 0;   // stones of the player TO MOVE
  uint64_t mask = 0;  // all stones
  uint8_t n = 0;      // plies played
  uint8_t won = 0;    // 1 if the mover just completed a line of K
};

struct Game {
  using State = ::game_demo_board::State;
  static constexpr int kActionCount = N;        // Move value == cell index y*W + x
  static constexpr int kMaxMoves = N;
  static constexpr int kMaxGameLength = N + 1;
  static constexpr int kInputPlanes = 2, kInputH = H, kInputW = W;
  static constexpr int kNumSymmetries = 8;      // D4: {transpose, flipx, flipy} x combinations

  static State initial() { return State{}; }

  static int legal_moves(const State& s, Move* out) {
    // Othello/other family games: replace this body with your move generation.
    int k = 0;
    for (int c = 0; c < N; c++) if (!(s.mask >> c & 1)) out[k++] = (Move)c;
    return k;
  }

  // Win test LOCAL to the placed cell: <= K-2 cells probed per direction.
  static bool wins_from(const State& s, uint64_t mine, int cell) {
    const int x = cell % W, y = cell / W;
    const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
    for (auto& d : dirs) {
      int run = 1;
      for (int step = 1; step < K; step++) {
        int xx = x + d[0] * step, yy = y + d[1] * step;
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) break;
        if (mine >> (yy * W + xx) & 1) run++;
        else break;
      }
      for (int step = 1; step < K; step++) {
        int xx = x - d[0] * step, yy = y - d[1] * step;
        if (xx < 0 || xx >= W || yy < 0 || yy >= H) break;
        if (mine >> (yy * W + xx) & 1) run++;
        else break;
      }
      if (run >= K) return true;
    }
    return false;
  }

  static void apply(State& s, Move c) {
    uint64_t b = 1ULL << c;
    uint64_t mover = s.cur | b;                       // stones of the player who moved
    s.won = wins_from(s, mover, c) ? 1 : 0;
    s.cur ^= s.mask;                                  // ownership flips to the other player
    s.mask |= b;
    s.n++;
  }

  static bool is_terminal(const State& s) { return s.won || s.n == N; }
  static int current_player(const State& s) { return s.n & 1; }
  static float outcome(const State& s, int p) {
    if (!s.won) return 0.f;                            // board full with no line = draw
    int winner = (s.n - 1) & 1;
    return p == winner ? 1.f : -1.f;
  }

  static gai::Hash hash(const State& s) { return gai::mix64(s.cur ^ gai::mix64(s.mask)); }

  // plane 0 = stones of the player to move, plane 1 = opponent; row y is written as-is.
  static void encode(const State& s, float* out) {
    uint64_t mine = s.cur, theirs = s.cur ^ s.mask;
    for (int p = 0; p < 2; p++) {
      uint64_t bb = p == 0 ? mine : theirs;
      float* base = out + (size_t)p * N;
      for (int c = 0; c < N; c++) base[c] = (bb >> c & 1) ? 1.f : 0.f;
    }
  }

  // ---- D4 symmetries (only valid because W == H; set kNumSymmetries = 1 if dims differ) ----
  // Each symmetry s is a (transpose, flip_x, flip_y) triple applied to board coordinates.
  static inline int sym_cell(int sym, int cell) {
    int x = cell % W, y = cell / W;
    if (sym & 2) x = W - 1 - x;
    if (sym & 4) y = H - 1 - y;
    if (sym & 1) { int t = x; x = y; y = t; }
    return y * W + x;
  }
  static Move transform_move(Move m, int sym) { return (Move)sym_cell(sym, (int)m); }
  static void symmetry_input(const float* in, float* out, int sym) {
    for (int p = 0; p < kInputPlanes; p++)
      for (int c = 0; c < N; c++)
        out[p * N + sym_cell(sym, c)] = in[p * N + c];
  }

  static std::string to_string(const State& s) {
    uint64_t p0 = (s.n & 1) ? (s.cur ^ s.mask) : s.cur;  // player 0 stones
    std::string t;
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        uint64_t b = 1ULL << (y * W + x);
        t += (s.mask & b) ? ((p0 & b) ? 'X' : 'O') : '.';
      }
      t += '\n';
    }
    t += "move = cell index, row*8+col (row 0 on top): 0..63\n";
    return t;
  }
};
}  // namespace game_demo_board
