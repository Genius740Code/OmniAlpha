// Slow, obviously-correct implementation used for differential testing against Game. Written differently ON PURPOSE
// (here: history vector instead of a counter) so shared bugs are unlikely.
#pragma once
#include <vector>
#include "engine/core/api.hpp"
namespace game_template {
struct RefState { std::vector<int> taken; };
struct RefGame {
  using State = RefState;
  static int pile(const State& s) { int p = 21; for (int t : s.taken) p -= t; return p; }
  static State initial() { return {}; }
  static int legal_moves(const State& s, gai::Move* out) { int n = 0, p = pile(s); for (int k = 1; k <= 3; k++) if (k <= p) out[n++] = (gai::Move)(k - 1); return n; }
  static void apply(State& s, gai::Move m) { s.taken.push_back(m + 1); }
  static bool is_terminal(const State& s) { return pile(s) == 0; }
  static int current_player(const State& s) { return (int)(s.taken.size() % 2); }
  static float outcome(const State& s, int p) { if (!is_terminal(s)) return 0.f; int winner = (int)((s.taken.size() - 1) % 2); return p == winner ? 1.f : -1.f; }
};
}  // namespace game_template
