// Unit tests for the serve-side EnsembleEvaluator + its `ensemble:` spec parsing.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include "games/connect4/game.hpp"
#include "tools/common.hpp"
using namespace gai;
#define EXPECT(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)

// Constant-value evaluator that always prefers move 0; lets us check the mean/blend math.
template <class G> struct ConstEval : Evaluator<G> {
  float v; explicit ConstEval(float vv) : v(vv) {}
  void evaluate(const typename G::State* st, EvalResult<G>* o, int c) override {
    Move mv[G::kMaxMoves];
    for (int i = 0; i < c; i++) { int n = G::legal_moves(st[i], mv); o[i].n = n; o[i].value = v;
      for (int j = 0; j < n; j++) o[i].priors[j] = (j == 0 ? 1.f : 0.f); }
  }
};

int main() {
  using G = game_connect4::Game;
  auto s = G::initial();
  {
    // Two uniform members: priors must stay a valid distribution, value stays 0.
    std::vector<std::unique_ptr<Evaluator<G>>> mv;
    mv.push_back(std::make_unique<UniformEvaluator<G>>());
    mv.push_back(std::make_unique<UniformEvaluator<G>>());
    EnsembleEvaluator<G> ens(std::move(mv));
    EXPECT(ens.size() == 2);
    EvalResult<G> out[1]; ens.evaluate(&s, out, 1);
    EXPECT(out[0].n == 7);
    float sum = 0; for (int j = 0; j < out[0].n; j++) sum += out[0].priors[j];
    EXPECT(std::fabs(sum - 1.f) < 1e-5f);
    EXPECT(std::fabs(out[0].value) < 1e-6f);
  }
  {
    // Opposite value members average to 0; both agree on move 0 so its prior stays 1.
    std::vector<std::unique_ptr<Evaluator<G>>> mv;
    mv.push_back(std::make_unique<ConstEval<G>>(1.f));
    mv.push_back(std::make_unique<ConstEval<G>>(-1.f));
    EnsembleEvaluator<G> ens(std::move(mv));
    EvalResult<G> out[1]; ens.evaluate(&s, out, 1);
    EXPECT(std::fabs(out[0].value) < 1e-6f);
    EXPECT(std::fabs(out[0].priors[0] - 1.f) < 1e-5f);
    EXPECT(std::fabs(out[0].priors[1]) < 1e-6f);
  }
  {
    // Priors are blended then renormalized: 3 members, only the first likes move 1.
    std::vector<std::unique_ptr<Evaluator<G>>> mv;
    struct Mix : Evaluator<G> {
      float v; int hot; Mix(float vv, int h) : v(vv), hot(h) {}
      void evaluate(const typename G::State* st, EvalResult<G>* o, int c) override {
        Move m[G::kMaxMoves];
        for (int i = 0; i < c; i++) { int n = G::legal_moves(st[i], m); o[i].n = n; o[i].value = v;
          for (int j = 0; j < n; j++) o[i].priors[j] = (j == hot ? 1.f : 0.f); }
      }
    };
    mv.push_back(std::make_unique<Mix>(0.f, 1)); mv.push_back(std::make_unique<Mix>(0.f, 0)); mv.push_back(std::make_unique<Mix>(0.f, 0));
    EnsembleEvaluator<G> ens(std::move(mv));
    EvalResult<G> out[1]; ens.evaluate(&s, out, 1);
    float sum = 0; for (int j = 0; j < out[0].n; j++) sum += out[0].priors[j];
    EXPECT(std::fabs(sum - 1.f) < 1e-5f);
    EXPECT(std::fabs(out[0].priors[0] - (2.f / 3.f)) < 1e-5f);   // 2 of 3 members like move 0
    EXPECT(std::fabs(out[0].priors[1] - (1.f / 3.f)) < 1e-5f);
    EXPECT(out[0].priors[2] < 1e-6f);
  }
  {
    // Spec parsing: '+' separates members; a single member collapses to that evaluator.
    auto ev = make_evaluator<G>("ensemble:uniform+uniform", 3);
    EvalResult<G> out[2]; ev->evaluate(&s, out, 2);
    EXPECT(out[1].n == 7);
    float sum = 0; for (int j = 0; j < out[0].n; j++) sum += out[0].priors[j];
    EXPECT(std::fabs(sum - 1.f) < 1e-5f);
    auto ev1 = make_evaluator<G>("ensemble:uniform", 3);
    EvalResult<G> o1[1]; ev1->evaluate(&s, o1, 1); EXPECT(o1[0].n == 7);
    // cache wrapping the ensemble spec must not misparse the cap.
    auto evc = make_evaluator<G>("cache:256:ensemble:uniform+rollout", 3);
    EvalResult<G> o2[1]; evc->evaluate(&s, o2, 1); EXPECT(o2[0].n == 7);
  }
  std::printf("ensemble OK\n");
  return 0;
}
