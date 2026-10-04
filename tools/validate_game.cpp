// ./build/validate_game --game NAME [--games N] [--seed S]
// Checks the Game API contract on random playouts + differential test vs reference_game.hpp + undo + symmetries.
#include <algorithm>
#include <cmath>
#include <vector>
#include "tools/common.hpp"
using namespace gai;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { if (g_fail++ < 20) { std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); } } } while (0)

template <class T> int run(const Args& a) {
  using G = typename T::Game; using R = typename T::Ref;
  static_assert(GameLike<G>, "Game does not satisfy gai::GameLike (see engine/core/api.hpp)");
  int games = (int)a.num("games", 2000); Rng rng((uint64_t)a.num("seed", 12345));
  const int IN = input_size<G>();
  std::vector<float> e1(IN), e2(IN), e3(IN);
  long wins[2] = {0, 0}, draws = 0, plies = 0; int maxlen = 0; bool non_alternating = false; size_t max_branch = 0;

  { auto s = G::initial(); Move mv[G::kMaxMoves];
    CHECK(!G::is_terminal(s), "initial state is terminal");
    CHECK(G::legal_moves(s, mv) > 0, "no legal moves at start");
    CHECK(G::current_player(s) == 0, "player 0 should move first"); }

  for (int g = 0; g < games; g++) {
    auto s = G::initial(); [[maybe_unused]] typename std::conditional_t<std::is_void_v<R>, int, typename std::conditional_t<std::is_void_v<R>, G, R>::State> r{};
    if constexpr (!std::is_void_v<R>) r = R::initial();
    std::vector<Move> hist; int ply = 0;
    while (!G::is_terminal(s)) {
      CHECK(ply < G::kMaxGameLength, "game exceeds kMaxGameLength (infinite game?)"); if (ply >= G::kMaxGameLength) break;
      Move mv[G::kMaxMoves + 1]; int n = G::legal_moves(s, mv);
      CHECK(n > 0 && n <= G::kMaxMoves, "legal move count %d out of [1,%d]", n, G::kMaxMoves); if (n <= 0 || n > G::kMaxMoves) break;
      max_branch = std::max<size_t>(max_branch, n);
      for (int i = 0; i < n; i++) { CHECK(mv[i] >= 0 && mv[i] < G::kActionCount, "move %d outside [0,kActionCount)", mv[i]); for (int j = i + 1; j < n; j++) CHECK(mv[i] != mv[j], "duplicate legal move"); }
      int p = G::current_player(s); CHECK(p == 0 || p == 1, "bad player %d", p);
      // determinism + value semantics
      G::encode(s, e1.data()); G::encode(s, e2.data());
      CHECK(e1 == e2, "encode not deterministic");
      for (float x : e1) CHECK(std::isfinite(x), "non-finite encoding");
      auto copy = s; CHECK(G::hash(copy) == G::hash(s), "hash differs for identical state");
      Move m = mv[rng.below(n)];
      // apply on a copy must not mutate the original
      { auto t = s; G::apply(t, m); G::encode(s, e3.data()); CHECK(e1 == e3, "state corruption: apply on copy changed original"); 
        if constexpr (HasUndo<G>) { auto u = t; G::undo(u, m); G::encode(u, e3.data()); CHECK(e1 == e3 && G::hash(u) == G::hash(s) && G::is_terminal(u) == G::is_terminal(s) && G::current_player(u) == p, "undo does not restore state"); } }
      // differential vs reference
      if constexpr (!std::is_void_v<R>) {
        Move rm[G::kMaxMoves + 8]; int rn = R::legal_moves(r, rm);
        std::vector<Move> A(mv, mv + n), B(rm, rm + std::min(rn, G::kMaxMoves + 8)); std::sort(A.begin(), A.end()); std::sort(B.begin(), B.end());
        CHECK(A == B, "legal moves differ from reference at ply %d", ply);
        CHECK(R::current_player(r) == p, "current_player differs from reference");
        CHECK(R::is_terminal(r) == false, "reference says terminal but optimized does not");
        R::apply(r, m);
      }
      G::apply(s, m); hist.push_back(m); ply++;
      if constexpr (!std::is_void_v<R>) {
        CHECK(R::is_terminal(r) == G::is_terminal(s), "terminal status differs from reference after %s", G::to_string(s).c_str());
        if (R::is_terminal(r) && G::is_terminal(s)) for (int pl = 0; pl < 2; pl++) CHECK(R::outcome(r, pl) == G::outcome(s, pl), "outcome differs from reference");
      }
      if (!G::is_terminal(s) && G::current_player(s) == p) non_alternating = true;
    }
    if (G::is_terminal(s)) {
      float o0 = G::outcome(s, 0), o1 = G::outcome(s, 1);
      CHECK(o0 == -o1, "outcome not zero-sum (%g, %g)", o0, o1);
      CHECK(o0 == 1 || o0 == 0 || o0 == -1, "outcome must be +1/0/-1, got %g", o0);
      if (o0 > 0) wins[0]++; else if (o0 < 0) wins[1]++; else draws++;
    }
    plies += ply; maxlen = std::max(maxlen, ply);
    // symmetry check: replaying the transformed game must give the transformed encoding
    if constexpr (G::kNumSymmetries > 1) {
      if (g < 200) for (int sym = 1; sym < G::kNumSymmetries; sym++) {
        auto a0 = G::initial(), a1 = G::initial();
        for (Move m : hist) {
          G::encode(a0, e1.data()); G::encode(a1, e2.data()); G::symmetry_input(e1.data(), e3.data(), sym);
          CHECK(e2 == e3, "symmetry %d: encode(transformed state) != symmetry_input(encode(state))", sym);
          Move t = G::transform_move(m, sym); CHECK(t >= 0 && t < G::kActionCount, "transform_move out of range");
          G::apply(a0, m); G::apply(a1, t);
        }
        CHECK(G::is_terminal(a0) == G::is_terminal(a1) && G::outcome(a0, 0) == G::outcome(a1, 0), "symmetry %d changes outcome", sym);
      }
    }
  }
  CHECK(wins[0] + wins[1] + draws > 0, "no game ever reached a terminal state");
  std::printf("{\"game\":\"%s\",\"games\":%d,\"failures\":%d,\"avg_plies\":%.2f,\"max_plies\":%d,\"max_branching\":%zu,"
              "\"p0_wins\":%ld,\"p1_wins\":%ld,\"draws\":%ld,\"reference\":%s,\"undo\":%s,\"symmetries\":%d,\"non_alternating_turns\":%s}\n",
              a.str("game").c_str(), games, g_fail, (double)plies / games, maxlen, max_branch, wins[0], wins[1], draws,
              std::is_void_v<R> ? "false" : "true", HasUndo<G> ? "true" : "false", G::kNumSymmetries, non_alternating ? "true" : "false");
  if (std::is_void_v<R>) std::printf("NOTE: no reference_game.hpp -> differential test skipped\n");
  std::printf(g_fail ? "VALIDATION FAILED (%d)\n" : "VALIDATION PASSED\n", g_fail);
  return g_fail ? 1 : 0;
}

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) { std::fprintf(stderr, "usage: validate_game --game NAME [--games N] [--seed S]\n"); return 2; }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
