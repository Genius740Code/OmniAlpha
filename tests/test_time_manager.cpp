#include <cstdio>
#include <cstdlib>
#include "engine/time_manager/time_manager.hpp"
using namespace gai;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
int main() {
  ClockConfig c; c.minimum_think_ms = 4000; c.maximum_think_ms = 30000; c.safety_margin_ms = 100; c.emergency_ms = 1000;
  // min_think must never cause a timeout
  for (double rem : {50.0, 100.0, 500.0, 900.0, 1500.0, 3000.0, 6000.0, 60000.0}) {
    ClockState s; s.remaining_ms = rem; s.increment_ms = 0; auto a = allocate_time(c, s, 2.5);
    EXPECT(a.hard_ms <= std::max(0.0, rem - c.safety_margin_ms) + 1e-9); EXPECT(a.soft_ms <= a.hard_ms + 1e-9); EXPECT(a.soft_ms >= 0);
    EXPECT(a.soft_ms <= 0.5 * (rem - c.safety_margin_ms) + 1e-9 || rem <= c.safety_margin_ms);
  }
  { ClockState s; s.remaining_ms = 800; auto a = allocate_time(c, s); EXPECT(a.emergency); EXPECT(a.soft_ms < 200); }
  { ClockState s; s.remaining_ms = 80; auto a = allocate_time(c, s); EXPECT(a.emergency && a.soft_ms == 0 && a.hard_ms == 0); }
  // increment buys more time; complexity scales time; max cap respected
  { ClockConfig d; ClockState s0, s1; s0.remaining_ms = s1.remaining_ms = 60000; s1.increment_ms = 2000;
    EXPECT(allocate_time(d, s1).soft_ms > allocate_time(d, s0).soft_ms);
    EXPECT(allocate_time(d, s0, 2.0).soft_ms > allocate_time(d, s0, 0.5).soft_ms);
    d.maximum_think_ms = 500; EXPECT(allocate_time(d, s1, 2.5).hard_ms <= 500); }
  std::printf("time_manager OK\n"); return 0;
}
