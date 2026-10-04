// ./build/benchmark_game --game NAME [--seconds 0.5]   -> one JSON line (profiler input for config selection)
#include <chrono>
#include <vector>
#include "tools/common.hpp"
using namespace gai;
using clk = std::chrono::steady_clock;
static volatile uint64_t g_sink;

template <class F> double rate(double secs, F&& f) {  // f(batch) performs `batch` ops, returns ops/sec
  auto t0 = clk::now(); long ops = 0;
  while (std::chrono::duration<double>(clk::now() - t0).count() < secs) ops += f(256);
  return ops / std::chrono::duration<double>(clk::now() - t0).count();
}

template <class T> int run(const Args& a) {
  using G = typename T::Game; double secs = a.dbl("seconds", 0.5); Rng rng(7);
  // collect a pool of realistic states from random playouts
  std::vector<typename G::State> pool; double plies = 0, branch = 0, nb = 0; int games = 0;
  while (pool.size() < 4096 && games < 2000) {
    auto s = G::initial(); int ply = 0;
    while (!G::is_terminal(s)) { Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv); branch += n; nb++; pool.push_back(s); G::apply(s, mv[rng.below(n)]); ply++; }
    plies += ply; games++;
  }
  size_t P = pool.size(); size_t idx = 0; std::vector<float> buf(input_size<G>());
  double r_legal = rate(secs, [&](int b) { Move mv[G::kMaxMoves]; int k = 0; for (int i = 0; i < b; i++) k += G::legal_moves(pool[idx++ % P], mv); g_sink = k; return b; });
  double r_apply = rate(secs, [&](int b) { uint64_t acc = 0; for (int i = 0; i < b; i++) { Move mv[G::kMaxMoves]; auto s = pool[idx++ % P]; if (G::is_terminal(s)) continue; G::legal_moves(s, mv); G::apply(s, mv[0]); acc += G::hash(s) & 1; } g_sink = acc; return b; });
  double r_term = rate(secs, [&](int b) { int k = 0; for (int i = 0; i < b; i++) k += G::is_terminal(pool[idx++ % P]); g_sink = k; return b; });
  double r_enc = rate(secs, [&](int b) { for (int i = 0; i < b; i++) G::encode(pool[idx++ % P], buf.data()); g_sink = (uint64_t)buf[0]; return b; });
  double r_hash = rate(secs, [&](int b) { uint64_t h = 0; for (int i = 0; i < b; i++) h ^= G::hash(pool[idx++ % P]); g_sink = h; return b; });
  long total_plies = 0;
  double r_play = rate(secs, [&](int b) { int g = 0; for (int i = 0; i < std::max(1, b / 64); i++, g++) { auto s = G::initial(); while (!G::is_terminal(s)) { Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv); G::apply(s, mv[rng.below(n)]); total_plies++; } } return g; });
  // MCTS throughput with the uniform evaluator (pure search overhead) and rollout evaluator, batch 1 and 16
  auto mcts_rate = [&](Evaluator<G>& ev, int batch) {
    return rate(secs, [&](int) { MCTS<G> m; m.set_root(G::initial()); m.run(512, ev, batch); return 512; });
  };
  UniformEvaluator<G> uni; RolloutEvaluator<G> roll(1, 3);
  double m_uni1 = mcts_rate(uni, 1), m_uni16 = mcts_rate(uni, 16), m_roll = mcts_rate(roll, 1);
  std::printf("{\"game\":\"%s\",\"state_bytes\":%zu,\"action_count\":%d,\"max_moves\":%d,\"input_size\":%d,\"symmetries\":%d,"
              "\"avg_game_plies\":%.1f,\"avg_branching\":%.2f,"
              "\"legal_moves_per_s\":%.3g,\"apply_per_s\":%.3g,\"terminal_per_s\":%.3g,\"encode_per_s\":%.3g,\"hash_per_s\":%.3g,"
              "\"random_games_per_s\":%.3g,\"mcts_sims_per_s_uniform_b1\":%.3g,\"mcts_sims_per_s_uniform_b16\":%.3g,\"mcts_sims_per_s_rollout\":%.3g}\n",
              a.str("game").c_str(), sizeof(typename G::State), G::kActionCount, G::kMaxMoves, input_size<G>(), G::kNumSymmetries,
              plies / games, branch / nb, r_legal, r_apply, r_term, r_enc, r_hash, r_play, m_uni1, m_uni16, m_roll);
  (void)total_plies; return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: benchmark_game --game NAME [--seconds S]\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
