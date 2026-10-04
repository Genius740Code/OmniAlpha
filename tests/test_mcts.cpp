#include <cstdio>
#include <cstdlib>
#include "games/connect4/game.hpp"
#include "games/template/game.hpp"
#include "engine/mcts/adaptive.hpp"
using namespace gai;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
int main() {
  { // Connect-4: player 0 has 3 stacked in column 0 -> must play column 0
    using G = game_connect4::Game; auto s = G::initial(); for (Move m : {0, 1, 0, 1, 0, 1}) G::apply(s, m);
    for (int batch : {1, 8}) { RolloutEvaluator<G> ev(1, 5); MCTS<G> t; t.set_root(s); t.run(600, ev, batch); EXPECT(t.best_move() == 0); EXPECT(t.root_value() > 0.5f); }
  }
  { // Connect-4: must block opponent's immediate win. P0: 0,0,0 stacked? build: P1 threatens column 1.
    using G = game_connect4::Game; auto s = G::initial(); for (Move m : {3, 1, 3, 1, 5, 1}) G::apply(s, m);  // P1 has 3 in col 1; P0 to move
    RolloutEvaluator<G> ev(1, 6); MCTS<G> t; t.set_root(s); t.run(1500, ev, 1); EXPECT(t.best_move() == 1);
  }
  { // Take-away: 21 stones, optimal first move takes 1 (leave 20). Terminal backup + tree reuse + compaction.
    using G = game_template::Game; UniformEvaluator<G> ev; MctsConfig c; c.max_nodes = 50;
    MCTS<G> t(c); t.set_root(G::initial()); t.run(3000, ev, 4); EXPECT(t.best_move() == 0);
    Move m = t.best_move(); t.advance_root(m); t.run(500, ev, 1); EXPECT(t.root_value() <= 1.f && t.root_value() >= -1.f);
    float pol[3]; t.visit_policy(pol); EXPECT(pol[0] + pol[1] + pol[2] > 0.99f);
  }
  { // adaptive search: obvious position stops early, honours max sims
    using G = game_connect4::Game; auto s = G::initial(); for (Move m : {0, 1, 0, 1, 0, 1}) G::apply(s, m);
    RolloutEvaluator<G> ev(1, 7); MCTS<G> t; t.set_root(s); SearchLimits L; L.min_simulations = 64; L.target_simulations = 2000; L.max_simulations = 4000;
    auto r = adaptive_search<G>(t, ev, L); EXPECT(r.best == 0); EXPECT(r.sims >= 64 && r.sims <= 4000);
    std::printf("adaptive: sims=%d early=%d extended=%d\n", r.sims, r.early_stop, r.extended);
  }
  std::printf("mcts OK\n"); return 0;
}
