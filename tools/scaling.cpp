// ./build/scaling --game NAME --evaluator-a SPEC --evaluator-b SPEC --sims-base N --doublings K --games G [--seed S]
// Arena: player A vs B with different simulation counts, alternating colors.
// Reports W/L/D, Elo of A over B and 95% CI as JSON per doubling pair.
// Pattern: A(N) vs A(2N) vs A(4N) vs ... vs A(2^K N), i.e. K pairs:
//   pair 0: A(N) vs A(2N), pair 1: A(2N) vs A(4N), ..., pair K-1: A(2^{K-1}N) vs A(2^K N)
#include "engine/evaluation/elo.hpp"
#include "tools/common.hpp"
using namespace gai;

template <class T> int run(const Args& a) {
  using G = typename T::Game;
  std::string evaluator_a_str = a.str("evaluator-a", "rollout");
  std::string evaluator_b_str = a.str("evaluator-b", "rollout");

  int games = (int)a.num("games", 10);
  int sims_base = (int)a.num("sims-base", 25);
  int doublings = (int)a.num("doublings", 3);
  uint64_t seed = (uint64_t)a.num("seed", 0);

  for (int d = 0; d < doublings; d++) {
    int sims_a = sims_base * (1 << d);
    int sims_b = sims_base * (1 << (d + 1));

    std::unique_ptr<Evaluator<G>> ev_a = make_evaluator<G>(evaluator_a_str, seed + (uint64_t)d * 1000);
    std::unique_ptr<Evaluator<G>> ev_b = make_evaluator<G>(evaluator_b_str, seed + (uint64_t)d * 1000 + 1);

    int w = 0, l = 0, d_count = 0;  // wins/losses/draws for A over B
    int opening_plies = (int)a.num("opening-plies", 2);

    Rng rng(seed + (uint64_t)d * 100000 + 1);

    for (int g = 0; g < games; g++) {
      int a_is = g & 1;  // which player id A takes (0 or 1), alternates per game
      auto s = G::initial(); Rng orng(1000 + g / 2);
      for (int i = 0; i < opening_plies && !G::is_terminal(s); i++) {
        Move mv[G::kMaxMoves];
        int n = G::legal_moves(s, mv);
        if (n == 0) break;
        G::apply(s, mv[orng.below(n)]);
      }

      MCTS<G> tree_a, tree_b;
      tree_a.set_root(s); tree_b.set_root(s);

      while (!G::is_terminal(s)) {
        int who = G::current_player(s);  // 0 or 1, the player to move
        SearchLimits L;
        L.min_simulations = L.target_simulations = L.max_simulations = (who == 0) ? sims_a : sims_b;
        auto& tree = (who == 0) ? tree_a : tree_b;
        Evaluator<G>* ev = (who == 0) ? ev_a.get() : ev_b.get();
        auto rep = adaptive_search<G>(tree, *ev, L);
        (void)rep;

        Move m;
        if (who == 0) {
          m = tree_a.sample_move(0.f, rng);
        } else {
          m = tree_b.sample_move(0.f, rng);
        }

        G::apply(s, m);
        tree_a.advance_root(m);
        tree_b.advance_root(m);
      }

      // Outcome from perspective of player a_is (the player A is assigned to in this game)
      float outcome_a = G::outcome(s, a_is);  // +1 if a_is won, -1 if lost, 0 draw
      if (outcome_a > 0) w++;
      else if (outcome_a < 0) l++;
      else d_count++;
    }

    auto e = estimate_elo(w, l, d_count);
    std::printf("{\"game\":\"%s\",\"doubling\":%d,\"sims_a\":%d,\"sims_b\":%d,\"games\":%d,\"wins\":%d,\"losses\":%d,\"draws\":%d,\"score\":%.3f,\"elo_a_over_b\":%.1f,\"elo_ci95\":%.1f}\n",
                a.str("game").c_str(), d, sims_a, sims_b, games, w, l, d_count, e.score, e.elo, e.ci95);
  }
  return 0;
}

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) {
    std::fprintf(stderr, "usage: scaling --game NAME [--evaluator-a SPEC] [--evaluator-b SPEC] [--sims-base N] [--doublings K] [--games G] [--seed S]\n");
    return 2;
  }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}