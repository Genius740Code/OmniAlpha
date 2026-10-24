// ./build/play --game NAME [--p0 human|mcts|random] [--p1 ...] [--sims N | --tc BASE+INC (seconds)]
//   [--model PATH] [--reload] [--cache [CAP]] (libtorch build: NN for the mcts player;
//   --reload hot-swaps the file each move when mtime changes AND the candidate model
//   actually loads -> play against the live 5-min training checkpoints; --cache wraps
//   the NN evaluator in a hash cache). [--solve-probe 1] runs the exact solver at the
//   root each move and plays a proven win immediately (else MCTS; budget --solve-probe-ms).
//   [--evaluator rollout] [--min-think-ms X] [--max-think-ms X] [--safety-ms X] [--emergency-ms X]
// With --tc the adaptive TimeManager + adaptive search drive thinking time; clocks are simulated from real elapsed time.
#include <chrono>
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include "tools/common.hpp"
#include "engine/evaluation/solver.hpp"
using namespace gai;

// The .ts file a evaluator spec ultimately serves ("" when the spec is not NN-based).
// Supports the serving-side spec forms: nn:<path>, nn-strict:<path>,
// hybrid:<path>:<alpha>[:<rollouts>] and cache[:<cap>]:<inner> (any of the above).
static std::string model_path_of(const std::string& spec) {
  if (spec.rfind("nn:", 0) == 0) return spec.substr(3);
  if (spec.rfind("nn-strict:", 0) == 0) return spec.substr(10);
  if (spec.rfind("hybrid:", 0) == 0) {
    std::string rest = spec.substr(7);
    auto c = rest.find(':');
    return c == std::string::npos ? rest : rest.substr(0, c);
  }
  if (spec.rfind("cache", 0) == 0 && spec.size() > 5 && spec[5] == ':') {
    std::string rest = spec.substr(6);
    auto numeric = [](const std::string& s) {
      return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
    };
    auto c_next = rest.find(':');  // strip the optional numeric capacity prefix
    if (c_next != std::string::npos && numeric(rest.substr(0, c_next))) rest = rest.substr(c_next + 1);
    if (rest.rfind("nn:", 0) == 0 || rest.rfind("nn-strict:", 0) == 0 ||
        rest.rfind("hybrid:", 0) == 0 || rest.rfind("cache", 0) == 0)
      return model_path_of(rest);
  }
  return "";
}

static long long file_mtime_ns(const std::string& path) {
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) return -1;
  return (long long)st.st_mtime * 1000000000ll + st.st_mtim.tv_nsec;
}

// Hot model reload for jam mode: re-create the evaluator when the model file mtime changes.
template <class G> long long current_model_mtime(const std::string& spec) {
  std::string path = model_path_of(spec);
  if (path.empty()) return -1;  // only NN-based specs support reload
  return file_mtime_ns(path);
}
template <class G> void maybe_reload(const Args& a, const std::string& spec,
                                     std::unique_ptr<Evaluator<G>>& ev, long long& last_mt) {
  if (!a.has("reload")) return;
  long long mt = current_model_mtime<G>(spec);
  if (mt < 0 || mt == last_mt) return;
  // Swap only AFTER the candidate model has actually loaded: a missing/corrupt .ts
  // keeps the previous evaluator (load errors are printed by NNEvaluator) and is
  // retried on the next move (last_mt intentionally left unchanged).
  auto cand = make_evaluator<G>(spec, 3);
  if (!cand->load_now()) {
    std::cout << "[model reload failed; keeping previous evaluator]\n";
    return;
  }
  last_mt = mt;
  ev = std::move(cand);
  std::cout << "[model reloaded]\n";
}



template <class T> int run(const Args& a) {
  using G = typename T::Game;
  std::string evaluator_str = a.str("evaluator", "rollout");

  // Support --model X.ts as a shortcut for nn:<path> (needs the libtorch build: build-lt)
  if (a.has("model")) {
    std::string model_path = a.str("model");
    evaluator_str = "nn:" + model_path;
  }
  // --cache [CAP]: wrap the NN evaluator in the hash-keyed eval cache
  // ("cache:<inner>" / "cache:<CAP>:<inner>" per the make_evaluator parser).
  // Reload replaces the WHOLE evaluator (cache included), so this does not
  // conflict with --reload: a cache per model generation is fine.
  bool nn_based = !model_path_of(evaluator_str).empty();
  if (a.has("cache") && nn_based) {
    std::string cap = a.str("cache");
    evaluator_str = (cap == "1" || cap.empty()) ? "cache:" + evaluator_str
                                                : "cache:" + cap + ":" + evaluator_str;
  }

  std::string kind[2] = {a.str("p0", "human"), a.str("p1", "mcts")};
  auto ev = make_evaluator<G>(evaluator_str, 3); Rng rng((uint64_t)a.num("seed", 9));
  // The mcts player is the only one that batches: when serving from an NN spec,
  // default the leaf batch to 32 instead of 1 (MCTS collects multi-leaf batches;
  // an explicit --batch always wins). Rollout/uniform specs keep batch 1.
  long long model_mtime = current_model_mtime<G>(evaluator_str);
  ClockConfig cc; cc.minimum_think_ms = a.dbl("min-think-ms", 0); cc.maximum_think_ms = a.dbl("max-think-ms", 30000);
  cc.safety_margin_ms = a.dbl("safety-ms", 100); cc.emergency_ms = a.dbl("emergency-ms", 1000);
  bool timed = a.has("tc"); double clock_ms[2] = {1e18, 1e18}, inc_ms = 0;
  if (timed) { double base = 0, inc = 0; std::sscanf(a.str("tc").c_str(), "%lf+%lf", &base, &inc); clock_ms[0] = clock_ms[1] = base * 1000; inc_ms = inc * 1000; }
  int solve_probe = (int)a.num("solve-probe", 0) != 0;
  long solve_probe_ms = (long)a.num("solve-probe-ms", 250);
  SearchLimits L; L.min_simulations = (int)a.num("min-sims", 200); L.target_simulations = (int)a.num("sims", 1000); L.max_simulations = (int)a.num("max-sims", 20000);
  L.batch = (int)a.num("batch", nn_based ? 32 : 1);
  float lcb_beta = (float)a.dbl("lcb", 0);
  MCTS<G> tree; auto s = G::initial(); tree.set_root(s); int ply = 0;
  while (!G::is_terminal(s)) {
    int p = G::current_player(s); std::cout << "\n" << G::to_string(s) << "player " << p << " (" << kind[p] << ")\n";
    Move m = -1; Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv);
    if (kind[p] == "human") {
      std::cout << "legal (" << n << "):"; for (int i = 0; i < n; i++) std::cout << " " << mv[i];
      std::cout << "\n";
      while (true) { std::cout << "move> " << std::flush; int x; if (!(std::cin >> x)) return 0; bool ok = false; for (int i = 0; i < n; i++) ok |= mv[i] == x; if (ok) { m = (Move)x; break; } std::cout << "illegal\n"; }
    } else if (kind[p] == "random") m = mv[rng.below(n)];
    else {
      maybe_reload(a, evaluator_str, ev, model_mtime);
      // Solver probe: before searching, ask the exact solver for a proven result
      // (budgeted iterative deepening; a non-zero value is game-theoretically
      // sound even at partial depth). Only a proven WIN for the mover short-
      // circuits the search; everything else falls through to MCTS.
      bool solved = false;
      if (solve_probe && !G::is_terminal(s)) {
        const auto sv = solve_state<G>(s, 0, solve_probe_ms);
        const int mover = G::current_player(s);
        const float vm = mover == 0 ? sv.value_from_p0 : -sv.value_from_p0;
        if (sv.exact && vm > 0.f && sv.best_move >= 0) {
          m = (Move)sv.best_move;
          std::cout << "solver: proven win for player " << mover << ", move " << m
                    << " (nodes " << sv.nodes << ", " << (int)(sv.seconds * 1000.0)
                    << " ms); skipping MCTS\n";
          solved = true;
        }
      }
      if (!solved) {
      SearchLimits l = L;
      if (timed) {
        ClockState cs; cs.remaining_ms = clock_ms[p]; cs.increment_ms = inc_ms; cs.est_moves_left = std::max(8, (G::kMaxGameLength - ply) / 2);
        auto al = allocate_time(cc, cs); l.soft_ms = al.soft_ms; l.hard_ms = al.hard_ms;
        if (al.emergency) { l.min_simulations = 1; l.target_simulations = 50; l.max_simulations = 200; }
        l.min_simulations = std::min(l.min_simulations, l.max_simulations);
      }
      auto t0 = std::chrono::steady_clock::now();
      if (!tree.root_expanded()) tree.run(1, *ev, 1);
      auto rep = adaptive_search<G>(tree, *ev, l);
      m = lcb_beta > 0 ? tree.best_move_lcb(lcb_beta) : rep.best;
      double used = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (timed) { clock_ms[p] = clock_ms[p] - used + inc_ms; if (clock_ms[p] <= 0) { std::cout << "player " << p << " LOST ON TIME\n"; return 0; } }
      std::cout << "mcts: move " << m << " sims " << rep.sims << " ms " << (int)used << " value " << rep.root_value << (rep.early_stop ? " [early]" : "") << (rep.extended ? " [extended]" : "") << "\n";
      }
    }
    G::apply(s, m); tree.advance_root(m); ply++;
  }
  std::cout << "\n" << G::to_string(s) << "result (player 0 view): " << G::outcome(s, 0) << "\n"; return 0;
}
int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: play --game NAME [--p0 human|mcts|random] [--p1 ...] [--sims N | --tc 60+1]\n"
    "  [--model PATH.ts] [--reload]   (libtorch build; reload = pick up new checkpoints live)\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
