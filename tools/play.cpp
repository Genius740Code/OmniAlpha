// ./build/play --game NAME [--p0 human|mcts|random] [--p1 ...] [--sims N | --tc BASE+INC (seconds)]
//   [--evaluator rollout] [--min-think-ms X] [--max-think-ms X] [--safety-ms X] [--emergency-ms X]
// With --tc the adaptive TimeManager + adaptive search drive thinking time; clocks are simulated from real elapsed time.
// Model files (--model) are a TODO (NN evaluator).
#include <chrono>
#include <iostream>
#include "tools/common.hpp"
using namespace gai;

template <class T> int run(const Args& a) {
  using G = typename T::Game;
  std::string evaluator_str = a.str("evaluator", "rollout");

  // Support --model X.pt as a shortcut for nn:<path>
  if (a.has("model")) {
    std::string model_path = a.str("model");
    evaluator_str = "nn:" + model_path;
  }

  std::string kind[2] = {a.str("p0", "human"), a.str("p1", "mcts")};
  auto ev = make_evaluator<G>(evaluator_str, 3); Rng rng((uint64_t)a.num("seed", 9));
  ClockConfig cc; cc.minimum_think_ms = a.dbl("min-think-ms", 0); cc.maximum_think_ms = a.dbl("max-think-ms", 30000);
  cc.safety_margin_ms = a.dbl("safety-ms", 100); cc.emergency_ms = a.dbl("emergency-ms", 1000);
  bool timed = a.has("tc"); double clock_ms[2] = {1e18, 1e18}, inc_ms = 0;
  if (timed) { double base = 0, inc = 0; std::sscanf(a.str("tc").c_str(), "%lf+%lf", &base, &inc); clock_ms[0] = clock_ms[1] = base * 1000; inc_ms = inc * 1000; }
  SearchLimits L; L.min_simulations = (int)a.num("min-sims", 200); L.target_simulations = (int)a.num("sims", 1000); L.max_simulations = (int)a.num("max-sims", 20000);
  L.batch = (int)a.num("batch", 1);
  MCTS<G> tree; auto s = G::initial(); tree.set_root(s); int ply = 0;
  while (!G::is_terminal(s)) {
    int p = G::current_player(s); std::cout << "\n" << G::to_string(s) << "player " << p << " (" << kind[p] << ")\n";
    Move m = -1; Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv);
    if (kind[p] == "human") {
      while (true) { std::cout << "move> " << std::flush; int x; if (!(std::cin >> x)) return 0; bool ok = false; for (int i = 0; i < n; i++) ok |= mv[i] == x; if (ok) { m = (Move)x; break; } std::cout << "illegal\n"; }
    } else if (kind[p] == "random") m = mv[rng.below(n)];
    else {
      SearchLimits l = L;
      if (timed) {
        ClockState cs; cs.remaining_ms = clock_ms[p]; cs.increment_ms = inc_ms; cs.est_moves_left = std::max(8, (G::kMaxGameLength - ply) / 2);
        auto al = allocate_time(cc, cs); l.soft_ms = al.soft_ms; l.hard_ms = al.hard_ms;
        if (al.emergency) { l.min_simulations = 1; l.target_simulations = 50; l.max_simulations = 200; }
        l.min_simulations = std::min(l.min_simulations, l.max_simulations);
      }
      auto t0 = std::chrono::steady_clock::now();
      if (!tree.root_expanded()) tree.run(1, *ev, 1);
      auto rep = adaptive_search<G>(tree, *ev, l); m = rep.best;
      double used = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (timed) { clock_ms[p] = clock_ms[p] - used + inc_ms; if (clock_ms[p] <= 0) { std::cout << "player " << p << " LOST ON TIME\n"; return 0; } }
      std::cout << "mcts: move " << m << " sims " << rep.sims << " ms " << (int)used << " value " << rep.root_value << (rep.early_stop ? " [early]" : "") << (rep.extended ? " [extended]" : "") << "\n";
    }
    G::apply(s, m); tree.advance_root(m); ply++;
  }
  std::cout << "\n" << G::to_string(s) << "result (player 0 view): " << G::outcome(s, 0) << "\n"; return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: play --game NAME [--p0 human|mcts|random] [--p1 ...] [--sims N | --tc 60+1]\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
