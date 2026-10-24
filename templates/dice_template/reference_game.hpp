// Slow, obviously-correct race implementation for differential testing against Game.
// Different by design: a move-history vector and rolls recomputed by REPLAYING the
// history with the same public chain formula (state is re-derived from scratch each call).
#pragma once
#include <vector>
#include "engine/core/api.hpp"
namespace game_template {
inline uint64_t ref_mix(uint64_t z) {  // mirrors engine/core/rng.hpp; kept local on purpose
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
struct RefState { std::vector<int> steps;  // per-ply step k (0 = pass); array parity == mover
};
struct RefGame {
  using State = RefState;
  static uint64_t chain_after(const State& s) {
    uint64_t c = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < s.steps.size(); i++) c = ref_mix(c + 0x2545F4914F6CDD1DULL * (uint64_t)(s.steps[i] + 1));
    return c;
  }
  static int roll(const State& s) { return 1 + (int)((ref_mix(chain_after(s) * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL) >> 25) % 6); }
  static int pos_of(const State& s, int p) {  // replay physical result incl. bonus + capture
    int pos[2] = {0, 0};
    for (size_t i = 0; i < s.steps.size(); i++) {
      int me = i % 2, op = me ^ 1, k = s.steps[i];
      if (k > 0) {
        int px = pos[me] + k;
        if (px > 0 && px < 30 && px % 7 == 0) px += 2;
        if (px > 30) px = 30;
        pos[me] = px;
        if (pos[op] == pos[me] && pos[me] > 0) pos[op] = pos[op] > 3 ? pos[op] - 3 : 0;
      }
    }
    return pos[p];
  }
  static State initial() { return {}; }
  static int legal_moves(const State& s, gai::Move* out) {
    int r = roll(s), me = s.steps.size() % 2, n = 0;
    for (int k = 1; k <= r; k++) if (pos_of(s, me) + k <= 30) out[n++] = (gai::Move)k;
    if (n == 0) out[n++] = 0;
    return n;
  }
  static void apply(State& s, gai::Move m) { s.steps.push_back(m); }
  static bool is_terminal(const State& s) { return pos_of(s, 0) == 30 || pos_of(s, 1) == 30; }
  static int current_player(const State& s) { return (int)(s.steps.size() % 2); }
  static float outcome(const State& s, int p) {
    int winner = pos_of(s, 0) == 30 ? 0 : 1;
    return p == winner ? 1.f : -1.f;
  }
};
}  // namespace game_template
