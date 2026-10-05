// ./build/selfplay --game NAME --games N --sims S --out data.bin [--threads T] [--evaluator uniform|rollout[:N]] [--seed X]
//   [--batch B] [--dirichlet 0.3] [--temp-moves 10]       Prints one JSON summary line.
// Each thread runs independent games with its own tree (batching across games on a shared GPU queue is a TODO).
#include <atomic>
#include <chrono>
#include <thread>
#include "tools/common.hpp"
using namespace gai;

template <class T> int run(const Args& a) {
  using G = typename T::Game;
  int games = (int)a.num("games", 100); int threads = (int)a.num("threads", std::max(1u, std::thread::hardware_concurrency()));
  SelfPlayConfig cfg; cfg.simulations = (int)a.num("sims", 200); cfg.batch = (int)a.num("batch", 1);
  cfg.temperature_moves = (int)a.num("temp-moves", 10); cfg.mcts.dirichlet_alpha = (float)a.dbl("dirichlet", 0.3);
  cfg.mcts.c_puct = (float)a.dbl("cpuct", 1.5);
  cfg.enable_shortcuts = (int)a.num("shortcuts", 1) != 0;
  cfg.policy_pruning = (int)a.num("prune", 1) != 0;
  cfg.mcts.forced_playouts = (int)a.num("forced", 1) != 0;
  cfg.mcts.forced_k = (float)a.dbl("forced-k", 2.0);
  cfg.full_search_prob = (float)a.dbl("full-prob", 1.0);
  cfg.cheap_sim_fraction = (float)a.dbl("cheap-frac", 0.125);
  cfg.cheap_min_sims = (int)a.num("cheap-min", 10);
  cfg.mcts.gumbel = (int)a.num("gumbel", 0) != 0;
  cfg.mcts.gumbel_sims = (int)a.num("gumbel-sims", 32);
  cfg.value_lambda = (float)a.dbl("value-lambda", 0.1);
  cfg.mcts.prior_temp = (float)a.dbl("prior-temp", 1.0);
  cfg.resign_q = (float)a.dbl("resign-q", -2.0);
  cfg.resign_min_plies = (int)a.num("resign-min-plies", 10);
  std::string ev_spec = a.str("evaluator", "rollout"); uint64_t seed = (uint64_t)a.num("seed", 1);
  SampleWriter<G> writer(a.str("out", "selfplay.bin").c_str());
  if (!writer.ok()) { std::fprintf(stderr, "cannot open output\n"); return 1; }
  std::atomic<int> next{0}, done{0}; std::atomic<long> positions{0}, w0{0}, w1{0}, dr{0}; auto t0 = std::chrono::steady_clock::now();
  auto worker = [&](int tid) {
    Rng rng(seed * 1000003 + tid); auto ev = make_evaluator<G>(ev_spec, seed * 77 + tid);
    while (next.fetch_add(1) < games) {
      auto rec = play_selfplay_game<G>(cfg, *ev, rng); writer.write(rec);
      positions += (long)rec.samples.size(); done++;
      (rec.outcome_p0 > 0 ? w0 : rec.outcome_p0 < 0 ? w1 : dr)++;
    }
  };
  std::vector<std::thread> th; for (int i = 0; i < threads; i++) th.emplace_back(worker, i); for (auto& t : th) t.join();
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("{\"game\":\"%s\",\"games\":%d,\"threads\":%d,\"sims\":%d,\"positions\":%ld,\"seconds\":%.2f,\"games_per_s\":%.3f,"
              "\"positions_per_s\":%.1f,\"p0_wins\":%ld,\"p1_wins\":%ld,\"draws\":%ld}\n",
              a.str("game").c_str(), games, threads, cfg.simulations, positions.load(), secs, games / secs, positions / secs, w0.load(), w1.load(), dr.load());
  return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: selfplay --game NAME --games N --sims S --out FILE\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
