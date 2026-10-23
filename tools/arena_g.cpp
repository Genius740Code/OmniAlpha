// SCRATCH TOOL (agent B, search-experiments): copy of tools/evaluate.cpp with two
// extra per-agent knobs used for frozen-net search A/B on the position-suite ruler:
//   --fpu-a/--fpu-b F        FPU reduction for unvisited children (default 0.25)
//   --gumbel-a/--gumbel-b S  agent uses Gumbel root + Sequential Halving with root
//                            budget S sims (instead of adaptive_search/PUCB)
//   --gumbel-mode-a/b 0|1    sigma mode: 1 = mctx (default), 0 = spread-matched
//   --cvisit-a/--cvisit-b F, --cscale-a/--cscale-b F  (mctx sigma knobs, defaults 50/0.1)
// Everything else (openings, pairing, Elo output, tc, lcb, max-sims) is identical
// to evaluate.cpp so results are directly comparable.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "engine/evaluation/elo.hpp"
#include "engine/mcts/gumbel_root.hpp"
#include "tools/common.hpp"
using namespace gai;

static constexpr int kTcOnlySims = 2'000'000;

static void push_ms(std::vector<double>& v, std::chrono::steady_clock::time_point t0) {
  v.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

template <class T> int run(const Args& a) {
  using G = typename T::Game;
  std::string evaluator_a_str = a.str("evaluator-a", "rollout");
  std::string evaluator_b_str = a.str("evaluator-b", "rollout");
  if (a.has("model")) {
    std::string model_path = a.str("model");
    evaluator_a_str = "nn:" + model_path;
    evaluator_b_str = "nn:" + model_path;
  }

  int games = (int)a.num("games", 100);
  int tc[2] = {(int)std::max(0L, a.num("tc-a", a.num("tc-ms", 0))),
               (int)std::max(0L, a.num("tc-b", a.num("tc-ms", 0)))};
  int sims[2] = {(int)a.num("sims-a", tc[0] > 0 ? kTcOnlySims : 400),
                 (int)a.num("sims-b", tc[1] > 0 ? kTcOnlySims : 100)};
  float cpuct[2] = {(float)a.dbl("cpuct-a", 1.5), (float)a.dbl("cpuct-b", 1.5)};
  float fpu[2] = {(float)a.dbl("fpu-a", 0.25), (float)a.dbl("fpu-b", 0.25)};
  float lcb[2] = {(float)a.dbl("lcb-a", 0), (float)a.dbl("lcb-b", 0)};
  float ptemp[2] = {(float)a.dbl("prior-temp-a", 1.0), (float)a.dbl("prior-temp-b", 1.0)};
  int gumbel_sims[2] = {(int)a.num("gumbel-a", 0), (int)a.num("gumbel-b", 0)};
  int gmode[2] = {(int)a.num("gumbel-mode-a", 1), (int)a.num("gumbel-mode-b", 1)};
  float cvisit[2] = {(float)a.dbl("cvisit-a", 50.f), (float)a.dbl("cvisit-b", 50.f)};
  float cscale[2] = {(float)a.dbl("cscale-a", 0.1f), (float)a.dbl("cscale-b", 0.1f)};
  double safety[2] = {a.dbl("safety-a", 1.0), a.dbl("safety-b", 1.0)};
  int threads = std::max(1, (int)a.num("threads", 1));
  int opening_plies = (int)a.num("opening-plies", 2);
  MctsConfig mc[2];
  for (int i = 0; i < 2; i++) {
    mc[i].c_puct = cpuct[i];
    mc[i].fpu_reduction = fpu[i];
    mc[i].prior_temp = ptemp[i];
    mc[i].gumbel_sims = gumbel_sims[i];
    mc[i].gumbel_sigma_mctx = gmode[i] != 0;
    mc[i].gumbel_c_visit = cvisit[i];
    mc[i].gumbel_c_scale = cscale[i];
  }
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
  std::vector<std::vector<double>> ms_a(threads), ms_b(threads);
  auto worker = [&](int tid) {
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
      int a_is = g & 1;
      auto s = G::initial(); Rng orng((uint64_t)a.num("seed", 5) * 100003ull + 1000 + g / 2);
      for (int i = 0; i < opening_plies && !G::is_terminal(s); i++) { Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv); G::apply(s, mv[orng.below(n)]); }
      MCTS<G> tree[2] = {MCTS<G>(mc[0]), MCTS<G>(mc[1])};
      tree[0].set_root(s); tree[1].set_root(s);
      while (!G::is_terminal(s)) {
        int who = (G::current_player(s) == a_is) ? 0 : 1;
        auto mt0 = std::chrono::steady_clock::now();
        Move m = -1;
        if (gumbel_sims[who] > 0) {
          // Gumbel root: budget is mc[who].gumbel_sims (root sims). No Dirichlet
          // (Gumbel sample is the exploration), no LCB/temp (SH winner is the pick).
          auto gr = run_gumbel_root(tree[who], *ev[who], mc[who], lrng, 1);
          m = gr.best;
          push_ms(who == 0 ? ms_a[tid] : ms_b[tid], mt0);
        } else {
          SearchLimits L;
          if (tc[who] > 0) {
            L.min_simulations = 1;
            L.target_simulations = L.max_simulations = std::max(1, sims[who]);
            L.soft_ms = L.hard_ms = (double)tc[who];
            tree[who].set_deadline(std::chrono::steady_clock::now() + std::chrono::milliseconds(tc[who]));
          } else {
            tree[who].clear_deadline();
            L.min_simulations = L.target_simulations = sims[who];
            L.max_simulations = std::max(sims[who], (int)a.num("max-sims", 6 * sims[who]));
          }
          L.batch = 1;
          L.safety_factor = safety[who];
          if (noise[who]) {
            if (!tree[who].root_expanded()) tree[who].run(1, *ev[who], 1);
            tree[who].add_root_noise(lrng);
          }
          auto rep = adaptive_search<G>(tree[who], *ev[who], L); (void)rep;
          push_ms(who == 0 ? ms_a[tid] : ms_b[tid], mt0);
          m = lcb[who] > 0 ? tree[who].best_move_lcb(lcb[who]) : tree[who].sample_move(0.f, lrng);
        }
        if (m < 0) m = tree[who].best_move();
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
  std::printf("{\"game\":\"%s\",\"games\":%d,\"sims_a\":%d,\"sims_b\":%d,\"gumbel_a\":%d,\"gumbel_b\":%d,"
              "\"wins\":%d,\"losses\":%d,\"draws\":%d,\"score\":%.3f,\"elo_a_over_b\":%.1f,\"elo_ci95\":%.1f}\n",
              a.str("game").c_str(), games, sims[0], sims[1], gumbel_sims[0], gumbel_sims[1],
              W, Lc, D, e.score, e.elo, e.ci95);
  return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: arena_g --game NAME [--gumbel-a S|--gumbel-b S] ...\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
