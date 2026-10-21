// ./build/evaluate --game NAME [--games 100] [--sims-a 400] [--sims-b 100] [--evaluator-a rollout] [--evaluator-b rollout]
//   [--max-sims N] (early-stop headroom; default 6x sims) [--seed S] (opening set)
//   [--tc-ms N | --tc-a N --tc-b N] (wall-clock ms/move; --tc-ms applies to both, --tc-a/--tc-b override.
//       When a tc is set and --sims-X is not given, that agent's sim cap is raised so wall clock is the
//       binding budget; an explicit --sims-X stays a hard cap on top of the clock.)
//   [--dirichlet-a A] [--dirichlet-b B] (optional root noise, selfplay --dirichlet semantics:
//       value <= 0 = auto alpha 10/#root_children; absent = noise off, legacy behavior)
//   evaluators: uniform|rollout[:N]|nn:<path>|ensemble:<a>+<b>+...|cache[:CAP]:<inner>
// Arena: player A vs B (colors alternate, paired openings). Reports W/L/D, Elo of A over B and 95% CI as JSON,
// plus the configured tc values and measured per-move search time (mean + p95) per agent.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "engine/evaluation/elo.hpp"
#include "tools/common.hpp"
using namespace gai;

// Sim cap when a time control is set and no --sims-X is given: large enough that wall
// clock is the binding budget for realistic tc values (also bounds tree memory).
static constexpr int kTcOnlySims = 2'000'000;

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
  // Wall-clock time controls (ms per move): --tc-ms is the base for both agents,
  // --tc-a/--tc-b override per agent (asymmetric races at equal wall-clock).
  int tc[2] = {(int)std::max(0L, a.num("tc-a", a.num("tc-ms", 0))),
               (int)std::max(0L, a.num("tc-b", a.num("tc-ms", 0)))};
  // sims defaults: unchanged without a tc; raised so time binds when tc-only.
  int sims[2] = {(int)a.num("sims-a", tc[0] > 0 ? kTcOnlySims : 400),
                 (int)a.num("sims-b", tc[1] > 0 ? kTcOnlySims : 100)};
  float cpuct[2] = {(float)a.dbl("cpuct-a", 1.5), (float)a.dbl("cpuct-b", 1.5)};
  float lcb[2] = {(float)a.dbl("lcb-a", 0), (float)a.dbl("lcb-b", 0)};
  float ptemp[2] = {(float)a.dbl("prior-temp-a", 1.0), (float)a.dbl("prior-temp-b", 1.0)};
  int threads = std::max(1, (int)a.num("threads", 1));
  int opening_plies = (int)a.num("opening-plies", 2);
  MctsConfig mc[2]; mc[0].c_puct = cpuct[0]; mc[1].c_puct = cpuct[1];
  mc[0].prior_temp = ptemp[0]; mc[1].prior_temp = ptemp[1];
  // Optional per-agent root Dirichlet noise (mirrors selfplay --dirichlet: value <= 0
  // = auto alpha 10/#root_children). Absent flags leave noise off (legacy behavior).
  for (int i = 0; i < 2; i++) {
    const char* k = i == 0 ? "dirichlet-a" : "dirichlet-b";
    if (!a.has(k)) continue;
    double dv = a.dbl(k, 0.0);
    if (dv <= 0) { mc[i].dirichlet_auto = true; mc[i].dirichlet_alpha = 0.f; }
    else mc[i].dirichlet_alpha = (float)dv;
  }
  bool noise[2] = {mc[0].dirichlet_auto || mc[0].dirichlet_alpha > 0,
                   mc[1].dirichlet_auto || mc[1].dirichlet_alpha > 0};
  std::atomic<int> next{0}, w{0}, l{0}, d{0};
  std::vector<std::vector<double>> ms_a(threads), ms_b(threads);  // measured search ms per move
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
        SearchLimits L;
        if (tc[who] > 0) {
          // Equal-time mode: wall clock stops the search (MCTS deadline inside run()
          // + hard_ms here). --sims-X stays a hard cap; min is lowered to 1 so
          // proven-win/loss early-stops can fire before either limit.
          L.min_simulations = 1;
          L.target_simulations = L.max_simulations = std::max(1, sims[who]);
          L.soft_ms = L.hard_ms = (double)tc[who];
          tree[who].set_deadline(std::chrono::steady_clock::now() + std::chrono::milliseconds(tc[who]));
        } else {
          tree[who].clear_deadline();
          // Headroom above target so proven-win/loss + can't-be-overtaken early-stops can fire.
          L.min_simulations = L.target_simulations = sims[who];
          L.max_simulations = std::max(sims[who], (int)a.num("max-sims", 6 * sims[who]));
        }
        L.batch = 1;
        if (noise[who]) {
          if (!tree[who].root_expanded()) tree[who].run(1, *ev[who], 1);
          tree[who].add_root_noise(lrng);
        }
        auto mt0 = std::chrono::steady_clock::now();
        auto rep = adaptive_search<G>(tree[who], *ev[who], L); (void)rep;
        auto& tms = who == 0 ? ms_a[tid] : ms_b[tid];
        tms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mt0).count());
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
  // Per-agent measured ms/move: mean + p95 across all moves of all games/threads.
  auto mstats = [](const std::vector<std::vector<double>>& per_thread, double& mean, double& p95) {
    std::vector<double> all;
    for (auto& v : per_thread) all.insert(all.end(), v.begin(), v.end());
    if (all.empty()) { mean = p95 = 0; return; }
    double s = 0; for (double x : all) s += x; mean = s / (double)all.size();
    std::sort(all.begin(), all.end());
    size_t idx = (size_t)std::ceil(0.95 * (double)all.size()) - 1;  // nearest-rank
    if (idx >= all.size()) idx = all.size() - 1;
    p95 = all[idx];
  };
  double a_mean = 0, a_p95 = 0, b_mean = 0, b_p95 = 0;
  mstats(ms_a, a_mean, a_p95); mstats(ms_b, b_mean, b_p95);
  auto e = estimate_elo(W, Lc, D);
  std::printf("{\"game\":\"%s\",\"games\":%d,\"sims_a\":%d,\"sims_b\":%d,\"tc_ms_a\":%d,\"tc_ms_b\":%d,"
              "\"ms_move_a_mean\":%.3f,\"ms_move_a_p95\":%.3f,\"ms_move_b_mean\":%.3f,\"ms_move_b_p95\":%.3f,"
              "\"wins\":%d,\"losses\":%d,\"draws\":%d,\"score\":%.3f,\"elo_a_over_b\":%.1f,\"elo_ci95\":%.1f}\n",
              a.str("game").c_str(), games, sims[0], sims[1], tc[0], tc[1],
              a_mean, a_p95, b_mean, b_p95,
              W, Lc, D, e.score, e.elo, e.ci95);
  return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: evaluate --game NAME [--games N] [--sims-a N] [--sims-b N] [--tc-ms N | --tc-a N --tc-b N]\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
