// Gameplay clock management (separate from training budget). Nothing here is hard-coded to a game or time control:
// everything comes from ClockConfig. The search-side adaptivity lives in engine/mcts/adaptive.hpp.
#pragma once
#include <algorithm>

namespace gai {
struct ClockConfig {
  double minimum_think_ms = 0;      // soft floor; ALWAYS overridden by the safety rules below
  double maximum_think_ms = 30000;  // cap per move
  double safety_margin_ms = 100;    // never plan to use the last N ms (latency, GUI, GC...)
  double emergency_ms = 1000;       // below this remaining time -> fast mode
  double hard_multiple = 3.0;       // hard limit = soft * this (extension room for unstable positions)
};
struct ClockState {
  double remaining_ms = 60000;
  double increment_ms = 0;
  int est_moves_left = 30;          // caller's estimate of OWN moves left (game phase); clamped to >= 8
};
struct TimeAllocation {
  double soft_ms = 0;  // target think time
  double hard_ms = 0;  // absolute max for this move (search may extend towards it when unstable)
  bool emergency = false;
};

// complexity: ~1 normal, >1 spend more (uncertain/critical), <1 spend less. Clamped to [0.4, 2.5].
inline TimeAllocation allocate_time(const ClockConfig& c, const ClockState& s, double complexity = 1.0) {
  TimeAllocation a;
  double usable = std::max(0.0, s.remaining_ms - c.safety_margin_ms);
  if (usable <= 0) { a.emergency = true; return a; }  // 0 ms: play instantly
  if (s.remaining_ms < c.emergency_ms) {
    // Fast mode: ignore minimum_think entirely. Spend a small slice, bank the increment.
    a.emergency = true;
    a.soft_ms = std::min(usable * 0.04 + 0.5 * s.increment_ms, usable * 0.25);
    a.hard_ms = std::min(a.soft_ms * 2.0, usable * 0.3);
    return a;
  }
  complexity = std::clamp(complexity, 0.4, 2.5);
  double moves = std::max(8, s.est_moves_left);
  double base = usable / moves + 0.75 * s.increment_ms;
  double soft = std::clamp(base * complexity, c.minimum_think_ms, c.maximum_think_ms);
  double hard = std::min(c.maximum_think_ms, std::max(soft, soft * c.hard_multiple));
  // Safety caps override min_think: one move may never eat most of the remaining clock.
  soft = std::min(soft, usable * 0.5);
  hard = std::min(hard, usable * 0.6);
  hard = std::max(hard, soft);
  a.soft_ms = soft; a.hard_ms = hard;
  return a;
}
}  // namespace gai
