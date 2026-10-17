// ./build/evaluate --game NAME [--games 100] [--sims-a 400] [--sims-b 100] [--evaluator-a rollout] [--evaluator-b rollout]
//   [--max-sims N] (early-stop headroom; default 6x sims) [--seed S] (opening set)
//   evaluators: uniform|rollout[:N]|nn:<path>|ensemble:<a>+<b>+...|cache[:CAP]:<inner>
// Arena: player A vs B (colors alternate, paired openings). Reports W/L/D, Elo of A over B and 95% CI as JSON.
#include <atomic>
#include <thread>
#include <vector>
#include "engine/evaluation/elo.hpp"
#include "tools/common.hpp"
using namespace gai;

template <class T> int run(const Args& a) {
  using G = typename T::Game;
  std::string evaluator_a_str = a.str("evaluator-a", "rollout");
  std::string evaluator_b_str = a.str("evaluator-b", "rollout");

  // Support --model X.pt as a shortcut for nn:<path> for both players
  if (a.has("model")) {
    std::string model_path = a.str("model");
    evaluator_a_str = "nn:" + model_path;
    evaluator_b_str = "nn:" + model_path;
  }

  int games = (int)a.num("games", 100);
  int sims[2] = {(int)a.num("sims-a", 400), (int)a.num("sims-b", 100)};
  float cpuct[2] = {(float)a.dbl("cpuct-a", 1.5), (float)a.dbl("cpuct-b", 1.5)};
  float lcb[2] = {(float)a.dbl("lcb-a", 0), (float)a.dbl("lcb-b", 0)};
  float ptemp[2] = {(float)a.dbl("prior-temp-a", 1.0), (float)a.dbl("prior-temp-b", 1.0)};
  int threads = std::max(1, (int)a.num("threads", 1));
  int opening_plies = (int)a.num("opening-plies", 2);
  MctsConfig mc[2]; mc[0].c_puct = cpuct[0]; mc[1].c_puct = cpuct[1];
  mc[0].prior_temp = ptemp[0]; mc[1].prior_temp = ptemp[1];
  std::atomic<int> next{0}, w{0}, l{0}, d{0};
  auto worker = [&](int tid) {
    // One evaluator per thread: NN modules and RNGs are not shared across threads.
    // Cache-by-default (same rationale as selfplay): explicit cache: specs pass through.
    auto mk = [&](const std::string& spec, uint64_t sd) -> std::unique_ptr<Evaluator<G>> {
      if (spec.rfind("cache", 0) == 0 || spec == "uniform") return make_evaluator<G>(spec, sd);
      return std::unique_ptr<Evaluator<G>>(new CachingEvaluator<G>(make_evaluator<G>(spec, sd), 1u << 18));
    };
    auto e0 = mk(evaluator_a_str, 11 + (uint64_t)tid * 100003);
    auto e1 = mk(evaluator_b_str, 22 + (uint64_t)tid * 100003);
    std::unique_ptr<Evaluator<G>> ev[2] = {std::move(e0), std::move(e1)};
    Rng lrng((uint64_t)a.num("seed", 5) * 1000003 + (uint64_t)tid);
    while (true) {
      int g = next.fetch_add(1); if (g >= games) break;
      int a_is = g & 1;  // which player id A takes
      // randomized opening with paired colors: same opening (seed 1000+g/2) is reused for the
      // g and g^1 pair so each opening is played from both colors (variance reduction).
      // --seed offsets the opening set so independent arena batches (e.g. SPRT) are not
      // re-playing the same games when invoked repeatedly with the same --games.
      auto s = G::initial(); Rng orng((uint64_t)a.num("seed", 5) * 100003ull + 1000 + g / 2);
      for (int i = 0; i < opening_plies && !G::is_terminal(s); i++) { Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv); G::apply(s, mv[orng.below(n)]); }
      MCTS<G> tree[2] = {MCTS<G>(mc[0]), MCTS<G>(mc[1])};
      tree[0].set_root(s); tree[1].set_root(s);
      while (!G::is_terminal(s)) {
        int who = (G::current_player(s) == a_is) ? 0 : 1;  // 0 = A, 1 = B
        // Headroom above target so proven-win/loss + can't-be-overtaken early-stops can fire.
        SearchLimits L; L.min_simulations = L.target_simulations = sims[who];
        L.max_simulations = std::max(sims[who], (int)a.num("max-sims", 6 * sims[who]));
        L.batch = 1;
        auto rep = adaptive_search<G>(tree[who], *ev[who], L); (void)rep;
        Move m = lcb[who] > 0 ? tree[who].best_move_lcb(lcb[who]) : tree[who].sample_move(0.f, lrng);
        G::apply(s, m); tree[0].advance_root(m); tree[1].advance_root(m);
      }
      float o = G::outcome(s, a_is); if (o > 0) w++; else if (o < 0) l++; else d++;
    }
  };
  std::vector<std::thread> th;
  for (int t = 1; t < threads; t++) th.emplace_back(worker, t);
  worker(0);
  for (auto& t : th) t.join();
  int W = w, Lc = l, D = d;
  auto e = estimate_elo(W, Lc, D);
  std::printf("{\"game\":\"%s\",\"games\":%d,\"sims_a\":%d,\"sims_b\":%d,\"wins\":%d,\"losses\":%d,\"draws\":%d,\"score\":%.3f,\"elo_a_over_b\":%.1f,\"elo_ci95\":%.1f}\n",
              a.str("game").c_str(), games, sims[0], sims[1], W, Lc, D, e.score, e.elo, e.ci95);
  return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: evaluate --game NAME [--games N] [--sims-a N] [--sims-b N]\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
