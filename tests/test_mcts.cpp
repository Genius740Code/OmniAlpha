#include <cstdio>
#include <cstdlib>
#include "games/connect4/game.hpp"
#include "games/template/game.hpp"
#include "engine/mcts/adaptive.hpp"
#include "engine/selfplay/selfplay.hpp"
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
  { // advance_root must flip `proven` to the root's player-to-move convention.
    // Take-away: pile 8 is a proven LOSS for the player to move (every reply
    // leaves 7/6/5, each a proven win for the opponent). Descending flips the
    // view: pile 5 with the opponent to move is a proven WIN for them; pile 4
    // (the only non-losing reply) is a proven LOSS for whoever must move there.
    using G = game_template::Game; UniformEvaluator<G> ev;
    auto s = G::initial(); s.pile = 8;
    MCTS<G> t; t.set_root(s); t.run(2000, ev, 4);
    EXPECT(t.root_proven() == -1);    // solver proved the root lost for the mover
    size_t nodes = t.node_count();
    t.advance_root((Move)2);           // take 3 -> pile 5, opponent to move
    EXPECT(t.node_count() == nodes);   // advance keeps the subtree (no set_root reset)
    EXPECT(t.root_proven() == 1);     // root convention: proven WIN for player to move
    // adaptive_search must take the proven-WIN early stop ("play it now") rather
    // than the proven-loss/draw one, and pick the winning move.
    SearchLimits L; L.min_simulations = 8; L.target_simulations = 32; L.max_simulations = 64; L.batch = 4;
    auto r = adaptive_search<G>(t, ev, L);
    EXPECT(r.early_stop); EXPECT(t.root_proven() == 1); EXPECT(r.best == 0);
    t.advance_root((Move)0);           // take 1 -> pile 4: lost for the new mover
    EXPECT(t.root_proven() == -1);
    // Same scenario with compaction forced on advance_root (max_nodes < tree
    // size): compact() copies nodes verbatim, so the flipped root-convention
    // proven must survive it (checked: 134 nodes -> 26, proven still +1).
    MctsConfig cc; cc.max_nodes = 40;
    MCTS<G> t2(cc); t2.set_root(s); t2.run(2000, ev, 4);
    size_t before = t2.node_count();
    t2.advance_root((Move)2);
    EXPECT(t2.node_count() < before);  // compaction actually ran
    EXPECT(t2.root_proven() == 1);
    t2.advance_root((Move)0);
    EXPECT(t2.root_proven() == -1);
  }
  { // batch dedup: a fresh root picked `batch` times in the first batch is ONE
    // position -- evaluate once, fan the result out to every duplicate leaf.
    using G = game_template::Game;
    struct Counting final : Evaluator<G> {
      UniformEvaluator<G> u; long states = 0;
      void evaluate(const typename G::State* st, EvalResult<G>* out, int n) override { states += n; u.evaluate(st, out, n); }
    };
    Counting ce; MCTS<G> t; t.set_root(G::initial()); t.run(8, ce, 8);
    EXPECT(ce.states == 1);           // 8 identical root leaves -> 1 evaluation
    EXPECT(t.total_visits() == 8);    // every leaf still backed up
    EXPECT(t.root_expanded());
  }
  { // selfplay shortcuts + reuse_tree: shortcut moves must keep the live tree
    // (advance_root) and still emit the same value-only samples. Constants match
    // the pre-fix baseline: 17 plies, 17 samples, P0 win, one mid-game block-in-1
    // (pile 5) and one terminal win-in-1 (pile 3).
    using G = game_template::Game; UniformEvaluator<G> ev;
    SelfPlayConfig cfg; cfg.simulations = 32; cfg.temperature_moves = 0;
    cfg.reuse_tree = true; cfg.enable_shortcuts = true;
    Rng rng(1234); auto rec = play_selfplay_game<G>(cfg, ev, rng);
    EXPECT(rec.plies == 17); EXPECT(rec.samples.size() == 17u); EXPECT(rec.outcome_p0 == 1.f);
    int shortcut_samples = 0;
    for (auto& sm : rec.samples) if (sm.full_search == 0.f) {
      shortcut_samples++;
      float sum = 0; int ones = 0;
      for (float p : sm.policy) { sum += p; if (p == 1.f) ones++; }
      EXPECT(ones == 1); EXPECT(sum == 1.f);   // one-hot, value-only target
      EXPECT(sm.q == 0.f || sm.q == 1.f);      // block=neutral prior, win=certain
    }
    EXPECT(shortcut_samples == 2);
    Rng rng2(1234); auto rec2 = play_selfplay_game<G>(cfg, ev, rng2);  // determinism
    EXPECT(rec2.plies == rec.plies && rec2.samples.size() == rec.samples.size());
    // Regression pin of the reuse itself (sims 16): the position right after the
    // pile-5 block shortcut searches the carried subtree, so its root value
    // differs from the reuse-off run. Before shortcut reuse both were -0.625.
    cfg.simulations = 16;
    cfg.reuse_tree = true; Rng r3(1234); auto on = play_selfplay_game<G>(cfg, ev, r3);
    cfg.reuse_tree = false; Rng r4(1234); auto off = play_selfplay_game<G>(cfg, ev, r4);
    EXPECT(on.plies == 19 && on.samples.size() == 19u);
    EXPECT(off.plies == 19 && off.samples.size() == 19u);
    EXPECT(on.samples[on.samples.size() - 2].q == -0.6f);     // carried subtree
    EXPECT(off.samples[off.samples.size() - 2].q == -0.625f);  // fresh tree
  }
  { // resign probes are counted and scored (false-resign calibration surfaces in
    // GameRecord; the selfplay tool aggregates to JSON). Constants are
    // deterministic for this config (evaluator seed 77, selfplay rng seed 9).
    using G = game_connect4::Game;
    SelfPlayConfig cfg; cfg.simulations = 64; cfg.temperature_moves = 0;
    cfg.resign_q = -0.5f; cfg.resign_min_plies = 8; cfg.resign_probe = 1.0f;
    RolloutEvaluator<G> ev(1, 77); Rng rng(9);
    long probes = 0, wins = 0;
    for (int i = 0; i < 20; i++) {
      auto rec = play_selfplay_game<G>(cfg, ev, rng);
      probes += rec.resign_probes; wins += rec.resign_probe_wins;
    }
    EXPECT(probes == 14); EXPECT(wins == 1);  // one "hopeless" side came back to win
    RolloutEvaluator<G> ev2(1, 77); Rng rng2(9);  // lanes path counts them too
    long lprobes = 0;
    for (auto& r : play_selfplay_lanes<G>(cfg, ev2, rng2, 4)) lprobes += r.resign_probes;
    EXPECT(lprobes == 7);
    cfg.resign_probe = 0.0f;  // probe off: hopeless games resign, nothing is counted
    RolloutEvaluator<G> ev3(1, 77); Rng rng3(9);
    long p0 = 0, short_games = 0;
    for (int i = 0; i < 20; i++) {
      auto rec = play_selfplay_game<G>(cfg, ev3, rng3);
      p0 += rec.resign_probes; if (rec.plies < 42) short_games++;
    }
    EXPECT(p0 == 0); EXPECT(short_games == 18);
  }
  std::printf("mcts OK\n"); return 0;
}
