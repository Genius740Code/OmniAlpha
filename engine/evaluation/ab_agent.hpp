// Depth-limited, time-budgeted alpha-beta ARENA agent.
//
// This is NOT the exact solver: callers bound the lookahead with `depth` (and a wall-clock
// budget) to get a strong-but-fast classical opponent for arena play. It reuses the same
// negamax machinery as engine/evaluation/solver.hpp at limited horizon, with:
//   * root ordering heuristic: immediate wins first, then forced blocks (occupying the
//     opponent's winning landing square), then center bias;
//   * deterministic tie handling: fixed search order = heuristic priority, ties by
//     ascending move index - reruns with the same depth and a non-binding time budget
//     are fully reproducible;
//   * iterative deepening 1..depth; a completed depth is replaced only by a deeper
//     completed depth (or a partial depth that proved a win/loss);
//   * the root immediate-win check means a winning move is never missed, at any depth >= 1.
//
// A depth >= remaining plies makes the agent play perfectly (it degenerates to exact
// solving); depth >= 2 already blocks single immediate threats (they are proven losses).
#pragma once
#include <cstdlib>
#include "engine/evaluation/solver.hpp"

namespace gai {

template <class G> struct AbAgent {
  using State = typename G::State;

  struct LastInfo {
    int score_ptm = 0;      // last search value from the player-to-move view
    int value_from_p0 = 0;   // same, from player 0's view (+1/0/-1)
    int depth = 0;           // deepest completed depth (plies of lookahead)
    long nodes = 0;
    bool exact = false;      // true => the chosen move's value is game-theoretically proven
    double seconds = 0.0;
    int n_legal = 0;
  };
  LastInfo last;

  // Returns the chosen move (a policy index) or -1 if the state is terminal / no move.
  int select_move(const State& s, int depth, long time_budget_ms) {
    const auto t0 = std::chrono::steady_clock::now();
    last = LastInfo{};
    if (G::is_terminal(s)) return -1;
    Move mv[G::kMaxMoves];
    const int n = G::legal_moves(s, mv);
    last.n_legal = n;
    if (n <= 0) return -1;
    if (depth <= 0) {  // no search: ordering heuristic only (win > block > center)
      last.depth = 0;
      last.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return root_order_move(s, mv, n);
    }
    if constexpr (Connect4Bitboard<G>) {
      solver_detail::C4Solver& sv = solver_detail::c4_solver();
      sv.b.set(0, time_budget_ms);
      int order[solver_detail::C4Solver::W];
      for (int i = 0; i < solver_detail::C4Solver::W; i++) order[i] = -1;  // -1 = unused slot
      build_root_order(s, mv, n, order);  // fills n playable columns; solve_id skips -1/full
      const int maxd = depth < solver_detail::C4Solver::CELLS - s.n ? depth : solver_detail::C4Solver::CELLS - s.n;
      const auto out = sv.solve_id(s.cur, s.mask, s.n, order, maxd);
      last.score_ptm = out.v;
      last.value_from_p0 = (int)solver_detail::c4_p0_value(out.v, G::current_player(s));
      last.depth = out.depth; last.exact = out.exact; last.nodes = sv.b.nodes;
      sv.b.set(0, 0);
      last.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return out.col;
    } else {
      auto& gs = solver_detail::generic_solver<G>();
      gs.b.set(0, time_budget_ms);
      Move order[G::kMaxMoves];
      int n_order = 0;
      // generic ordering: moves that immediately win first, then ascending index
      int prio[G::kMaxMoves];
      for (int i = 0; i < n; i++) {
        State t = s; G::apply(t, mv[i]);
        prio[i] = (G::is_terminal(t) && G::outcome(t, G::current_player(t)) < 0.f) ? 1 : 0;
      }
      for (int want = 1; want >= 0; want--)
        for (int i = 0; i < n; i++)
          if (prio[i] == want) order[n_order++] = mv[i];
      const int maxd = depth < G::kMaxGameLength ? depth : G::kMaxGameLength;
      const auto out = gs.solve_root_id(s, order, n_order, maxd);
      last.score_ptm = out.v;
      const int p = G::current_player(s);
      last.value_from_p0 = p == 0 ? out.v : -out.v;
      last.depth = out.depth; last.exact = out.exact; last.nodes = gs.b.nodes;
      gs.b.set(0, 0);
      last.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return out.mv;
    }
  }

  // Heuristic move without any search (used when depth <= 0): immediate win > block > center.
  int root_order_move(const State& s, const Move* mv, int n) {
    int order[G::kMaxMoves > 7 ? G::kMaxMoves : 7];
    const int n_order = build_root_order(s, mv, n, order);
    return n_order > 0 ? order[0] : (n > 0 ? mv[0] : -1);
  }

 private:
  // Root move ordering: immediate wins first, then blocks, then center bias; ties by
  // ascending move index. Fills `order` with the legal columns and returns its length.
  int build_root_order(const State& s, const Move* mv, int n, int* order) {
    if constexpr (Connect4Bitboard<G>) {
      const uint64_t poss = solver_detail::C4Solver::possible(s.mask);
      const uint64_t my_win = solver_detail::C4Solver::winning(s.cur, s.mask) & poss;
      const uint64_t opp_win = solver_detail::C4Solver::winning(s.cur ^ s.mask, s.mask) & poss;
      int prio[G::kMaxMoves];
      for (int i = 0; i < n; i++) {
        const int c = mv[i];
        const uint64_t bit = poss & solver_detail::C4Solver::COL(c);
        prio[i] = 3 - (c > 3 ? c - 3 : 3 - c);  // center bias: 3,2,2,1,1,0,0
        if (my_win & bit) prio[i] += 100;       // our immediate winning move
        if (opp_win & bit) prio[i] += 50;        // block the opponent's immediate win
      }
      int k = 0;
      for (int want = 103; want >= 0 && k < n; want--)
        for (int i = 0; i < n; i++)
          if (prio[i] == want) order[k++] = mv[i];
      return k;
    } else {
      for (int i = 0; i < n; i++) order[i] = mv[i];
      return n;
    }
  }
};

}  // namespace gai
