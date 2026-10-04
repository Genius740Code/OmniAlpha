#pragma once
#include <cmath>
namespace gai {
struct EloEstimate { double elo = 0, ci95 = 0, score = 0; int w = 0, l = 0, d = 0; };
// Elo of A relative to B from W/L/D, with a normal-approximation 95% CI (trinomial variance).
inline EloEstimate estimate_elo(int w, int l, int d) {
  EloEstimate e; e.w = w; e.l = l; e.d = d; double n = w + l + d;
  if (n <= 0) return e;
  double s = (w + 0.5 * d) / n; e.score = s;
  auto elo = [](double p) { p = std::fmin(std::fmax(p, 1e-4), 1 - 1e-4); return -400.0 * std::log10(1.0 / p - 1.0); };
  e.elo = elo(s);
  double var = (w * std::pow(1 - s, 2) + l * std::pow(0 - s, 2) + d * std::pow(0.5 - s, 2)) / n;
  double se = std::sqrt(var / n);
  e.ci95 = (elo(s + 1.96 * se) - elo(s - 1.96 * se)) / 2.0;
  return e;
}
}  // namespace gai
