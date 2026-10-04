#pragma once
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include "engine/core/api.hpp"
#include "engine/mcts/adaptive.hpp"
#include "engine/selfplay/selfplay.hpp"
#include "engine/time_manager/time_manager.hpp"

#ifdef GAI_LIBTORCH
#include "engine/inference/nn_evaluator.hpp"
#endif

#include "games.inc"

struct Args {
  std::map<std::string, std::string> kv;
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
      std::string a = argv[i]; if (a.rfind("--", 0) != 0) continue; a = a.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv[a] = argv[++i]; else kv[a] = "1";
    }
  }
  bool has(const char* k) const { return kv.count(k) > 0; }
  std::string str(const char* k, const char* d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
  long num(const char* k, long d) const { auto it = kv.find(k); return it == kv.end() ? d : std::atol(it->second.c_str()); }
  double dbl(const char* k, double d) const { auto it = kv.find(k); return it == kv.end() ? d : std::atof(it->second.c_str()); }
};

template <class G, class R> struct GameTag { using Game = G; using Ref = R; };

// Calls f(GameTag<G,R>{}) for the game called `name` (compile-time dispatch; no virtuals in the hot path).
template <class F> int dispatch_game(const std::string& name, F&& f) {
#define GAI_TRY(n, G, R) if (name == #n) return f(GameTag<G, R>{});
  GAI_GAME_LIST(GAI_TRY)
#undef GAI_TRY
  std::fprintf(stderr, "unknown --game '%s'. available:", name.c_str());
#define GAI_PRINT(n, G, R) std::fprintf(stderr, " %s", #n);
  GAI_GAME_LIST(GAI_PRINT)
#undef GAI_PRINT
  std::fprintf(stderr, "\n");
  return 2;
}

// --evaluator uniform|rollout[:N]  (nn:<path> supported when GAI_LIBTORCH=ON)
template <gai::GameLike G> std::unique_ptr<gai::Evaluator<G>> make_evaluator(const std::string& spec, uint64_t seed) {
  if (spec == "uniform") return std::make_unique<gai::UniformEvaluator<G>>();
  if (spec.rfind("rollout", 0) == 0) { int n = spec.size() > 8 ? std::atoi(spec.c_str() + 8) : 1; return std::make_unique<gai::RolloutEvaluator<G>>(n > 0 ? n : 1, seed); }
#ifdef GAI_LIBTORCH
  if (spec.rfind("nn:", 0) == 0) {
    std::string path = spec.substr(3);
    return std::make_unique<gai::NNEvaluator<G>>(path);
  }
#endif
  std::fprintf(stderr, "evaluator '%s' not supported yet (NN evaluator is TODO, see docs/HANDOFF.md)\n", spec.c_str());
  std::exit(3);
}
