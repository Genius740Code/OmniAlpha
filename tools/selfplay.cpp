// ./build/selfplay --game NAME --games N --sims S --out data.bin [--threads T] [--seed X]
//   [--evaluator uniform|rollout[:N]|ensemble:<a>+<b>+...|cache[:CAP]:<inner>]
//   [--batch B] [--dirichlet 0.3] [--temp-moves 10] [--tactics2 1]  Prints one JSON summary line.
// --dirichlet <= 0 enables auto mode: root noise alpha = 10/#root_children per root.
// --tactics2 1 enables the 2-ply forced-win (win-in-3) root shortcut (default off).
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
  cfg.temperature_moves = (int)a.num("temp-moves", 10);
  double dirichlet = a.dbl("dirichlet", 0.3);
  cfg.mcts.dirichlet_auto = dirichlet <= 0.0;  // <= 0: auto alpha = 10/#root_children per root
  cfg.mcts.dirichlet_alpha = dirichlet > 0.0 ? (float)dirichlet : 0.0f;
  cfg.tactics2 = (int)a.num("tactics2", 0) != 0;
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
  cfg.mcts.gumbel_sigma_mctx = (int)a.num("gumbel-sigma-mctx", 0) != 0;
  cfg.value_lambda = (float)a.dbl("value-lambda", 0.1);
  cfg.mcts.prior_temp = (float)a.dbl("prior-temp", 1.0);
  cfg.mcts.loss_fallthrough = (int)a.num("loss-fallthrough", 1) != 0;
  cfg.resign_q = (float)a.dbl("resign-q", -2.0);
  cfg.resign_min_plies = (int)a.num("resign-min-plies", 10);
  cfg.resign_probe = (float)a.dbl("resign-probe", 0.0);
  cfg.reuse_tree = (int)a.num("reuse-tree", 1) != 0;
  cfg.lanes = std::max(1, (int)a.num("lanes", 1));
  if (cfg.lanes > 1 && cfg.mcts.gumbel) {
    std::fprintf(stderr, "lanes>1 with gumbel: falling back to per-game (Gumbel has own batching)\n");
    cfg.lanes = 1;
  }
  std::string ev_spec = a.str("evaluator", "rollout"); uint64_t seed = (uint64_t)a.num("seed", 1);
  SampleWriter<G> writer(a.str("out", "selfplay.bin").c_str());
  if (!writer.ok()) { std::fprintf(stderr, "cannot open output\n"); return 1; }
  std::atomic<int> next{0}, done{0}; std::atomic<long> positions{0}, w0{0}, w1{0}, dr{0}, tac2{0}; std::atomic<long> probes{0}, probe_wins{0}; auto t0 = std::chrono::steady_clock::now();
  auto worker = [&](int tid) {
    Rng rng(seed * 1000003 + tid);
    // Cache-by-default: transpositions + tree reuse hit often (Connect4 board-only hash);
    // per-worker cache so no locking; explicit cache: specs pass through untouched.
    auto mk = [&](const std::string& spec, uint64_t sd) {
      if (spec.rfind("cache", 0) == 0 || spec == "uniform") return make_evaluator<G>(spec, sd);
      return std::unique_ptr<Evaluator<G>>(new CachingEvaluator<G>(make_evaluator<G>(spec, sd), 1u << 18));
    };
    auto ev = mk(ev_spec, seed * 77 + tid);
    auto count = [&](const GameRecord<G>& rec) {
      writer.write(rec);
      positions += (long)rec.samples.size(); done++;
      tac2 += rec.tactic2_solves;
      probes += rec.resign_probes; probe_wins += rec.resign_probe_wins;
      if (rec.outcome_p0 > 0) w0++; else if (rec.outcome_p0 < 0) w1++; else dr++;
    };
    if (cfg.lanes > 1) {
      while (true) {
        int base = next.fetch_add(cfg.lanes);
        if (base >= games) break;
        int L = std::min(cfg.lanes, games - base);
        for (auto& rec : play_selfplay_lanes<G>(cfg, *ev, rng, L)) count(rec);
      }
      return;
    }
    while (next.fetch_add(1) < games) {
      auto rec = play_selfplay_game<G>(cfg, *ev, rng); count(rec);
    }
  };
  std::vector<std::thread> th; for (int i = 0; i < threads; i++) th.emplace_back(worker, i); for (auto& t : th) t.join();
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("{\"game\":\"%s\",\"games\":%d,\"threads\":%d,\"sims\":%d,\"positions\":%ld,\"seconds\":%.2f,\"games_per_s\":%.3f,"
              "\"positions_per_s\":%.1f,\"p0_wins\":%ld,\"p1_wins\":%ld,\"draws\":%ld,\"tactics2_solves\":%ld,"
              "\"resign_probes\":%ld,\"resign_probe_wins\":%ld,\"false_resign_rate\":%.4f}\n",
              a.str("game").c_str(), games, threads, cfg.simulations, positions.load(), secs, games / secs, positions / secs,
              w0.load(), w1.load(), dr.load(), tac2.load(), probes.load(), probe_wins.load(),
              probes.load() ? (double)probe_wins.load() / (double)probes.load() : 0.0);
  return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: selfplay --game NAME --games N --sims S --out FILE\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
