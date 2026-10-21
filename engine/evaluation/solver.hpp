// Exact game-theoretic solver + depth-limited alpha-beta machinery ("classical search"
// strength anchor for the AlphaZero pipeline).
//
// Two engines live here:
//
//  1. Connect-4 exact solver (Tromp / Pascal-Pons style), operating directly on the
//     bitboard State layout of games/connect4 (7 bits per column, bit 6 sentinel, bit =
//     col*7 + row). Alpha-beta negamax with:
//       - win-distance scoring from the player-to-move view (a win completed with the
//         T-th stone on the board scores floor((44-T)/2); losses symmetric; draws 0), so a
//         position's value is a pure function of its stones and TT entries are path-free;
//       - transposition table: direct-mapped (Pons-style: one cache line per probe,
//         home-slot replacement on miss), 2^25 slots. Slot placement is driven by
//         G::hash (mix64-based); the stored 64-bit tag is the injective bitboard identity
//         `cur + mask + 1` (depth-limited table: that tag | horizon << 49), so a slot hit
//         is an exact position match with zero collision risk.
//         Entry = { key, exact score or bound flag, best move }; the stored best move
//         also orders the children on re-visits (across the null-window passes);
//       - center-out move ordering: columns 4,3,5,2,6,1,0 (1-indexed), i.e. 3,2,4,1,5,0,6;
//       - immediate-win shortcut;
//       - forced-move shortcuts (Pons' possibleNonLosingMoves): unavoidable double threat
//         => proven loss; single opponent threat => forced block; "anticipated loser"
//         filtering of moves that hand the opponent the square directly above.
//     Value semantics follow the engine convention: +/-1/0 from the player-to-move view;
//     Solved::value_from_p0 is the same value rotated to player 0's viewpoint.
//
//  2. Generic fallback for any gai::GameLike game (e.g. games/template): negamax on the
//     G API, engine values +1/-1/0, TT keyed by G::hash, deterministic ascending move
//     order. Slower but universal - every registered game compiles.
//
// Budgets: solve_state(s, node_budget, time_budget_ms). Unlimited budgets run the direct
// exact solve (null-window binary search on the score). Limited budgets run top-level
// iterative deepening; the deepest *completed* depth is reported, values from partial
// depth are conservative hints (0 = "no proof within d plies") and Solved::exact=false.
// Any non-zero value is game-theoretically proven (it can only be backed by terminal /
// shortcut / fully-resolved lines), so exact=true for those even at partial depth - with
// one exception enforced by construction: a *partial* depth never reports a loss, because
// "all searched children lose" says nothing about the unsearched ones. Partial adoption
// is therefore limited to proven wins (exact=true) and 0 hints; losses require a fully
// completed depth.
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"

namespace gai {

// Result of solving one position. value_from_p0 is +1 (player 0 wins), 0 (draw),
// -1 (player 1 wins) under perfect play; best_move is for the player to move
// (-1 = terminal position / no move). exact=false => partial-depth conservative hint.
struct Solved {
  float value_from_p0 = 0.f;
  int best_move = -1;
  long nodes = 0;
  int depth_done = 0;
  bool exact = false;
  int score_ptm = 0;    // raw solver score from the player-to-move view (bitboard path:
                        // win-distance score; generic path: just -1/0/+1); diagnostics only.
  double seconds = 0.0;
};

namespace solver_detail {

#if defined(__GNUC__)
#define GAI_SOLVER_PREFETCH(p) __builtin_prefetch(p)
#else
#define GAI_SOLVER_PREFETCH(p) ((void)0)
#endif

struct TTEntry {  // 16 bytes, no padding
  uint64_t key;
  int32_t score;
  uint8_t flag;   // TT::kExact / kLower / kUpper; kEmpty marks an unused slot
  int8_t col;     // best move for the player to move (ordering hint), -1 = unknown
  uint16_t pad;
};
static_assert(sizeof(TTEntry) == 16, "packed 16-byte TT entry");

// Direct-mapped transposition table (Pons-style): exactly one cache line touched per
// probe; on a key miss the home slot is overwritten. The small collision loss versus
// open addressing is negligible next to the probe-latency win - the search is
// memory-bound at roughly one TT miss per visited node.
struct TT {
  static constexpr uint8_t kExact = 0, kLower = 1, kUpper = 2, kEmpty = 3;
  std::vector<TTEntry> slot;
  uint64_t mask;
  explicit TT(int bits) : slot((size_t)1 << bits, TTEntry{0, 0, kEmpty, -1, 0}),
                           mask(((uint64_t)1 << bits) - 1) {}
  size_t home(uint64_t key) const { return (size_t)(mix64(key) & mask); }
  bool probe_at(size_t i, uint64_t key, int32_t& score, uint8_t& flag, int8_t& col) const {
    const TTEntry& e = slot[i];
    if (e.key != key || e.flag == kEmpty) return false;  // col is only trusted on a hit
    score = e.score; flag = e.flag; col = e.col; return true;
  }
  void store_at(size_t i, uint64_t key, int32_t score, uint8_t flag, int8_t col) {
    TTEntry& e = slot[i];
    e.key = key; e.score = score; e.flag = flag; e.col = col; e.pad = 0;
  }
};

struct Budget {
  long nodes = 0, node_budget = 0;
  bool has_deadline = false, aborted = false;
  std::chrono::steady_clock::time_point deadline{};
  void set(long nb, long tb_ms) {
    nodes = 0; node_budget = nb; aborted = false; has_deadline = tb_ms > 0;
    if (has_deadline) deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(tb_ms);
  }
  inline bool stop() {
    if (aborted) return true;
    if (node_budget > 0 && nodes >= node_budget) { aborted = true; return true; }
    if (has_deadline && (nodes & 1023) == 0 && std::chrono::steady_clock::now() >= deadline) {
      aborted = true; return true;
    }
    return false;
  }
};

// ---------------------------------------------------------------------------
// Connect-4 core (bitboard layout of games/connect4: bit = col*7 + row, sentinel bit 6)
// ---------------------------------------------------------------------------
struct C4Core {
  static constexpr int W = 7, H = 6, CELLS = 42;
  static constexpr int ORDER[7] = {3, 2, 4, 1, 5, 0, 6};  // center-out
  static constexpr uint64_t BOTTOM_ALL = [] {
    uint64_t b = 0; for (int c = 0; c < W; c++) b |= 1ULL << (7 * c); return b;
  }();
  static constexpr uint64_t BOARD = BOTTOM_ALL * ((1ULL << H) - 1);  // all 42 cells
  static constexpr uint64_t COL(int c) { return 63ULL << (7 * c); }
  // landing squares of the next stone in every column
  static uint64_t possible(uint64_t mask) { return (mask + BOTTOM_ALL) & BOARD; }
  // all empty squares that would complete a 4-in-row for `pos` (Pascal Pons formulation)
  static uint64_t winning(uint64_t pos, uint64_t mask) {
    uint64_t r = (pos << 1) & (pos << 2) & (pos << 3);                      // vertical
    uint64_t p = (pos << 7) & (pos << 14);                                   // horizontal
    r |= p & (pos << 21);
    r |= p & (pos >> 7);
    p = (pos >> 7) & (pos >> 14);
    r |= p & (pos >> 21);
    r |= p & (pos << 7);
    p = (pos << 6) & (pos << 12);                                            // diagonal /
    r |= p & (pos << 18);
    r |= p & (pos >> 6);
    p = (pos >> 6) & (pos >> 12);
    r |= p & (pos >> 18);
    r |= p & (pos << 6);
    p = (pos << 8) & (pos << 16);                                            // anti-diagonal
    r |= p & (pos << 24);
    r |= p & (pos >> 8);
    p = (pos >> 8) & (pos >> 16);
    r |= p & (pos >> 24);
    r |= p & (pos << 8);
    return r & (BOARD ^ mask);
  }
  static int score_win_now(int n) { return (CELLS + 1 - n) / 2; }     // win with stone n+1
  static int score_lose_fast(int n) { return -(CELLS - n) / 2; }      // opponent wins at n+2
};

struct C4Solver : C4Core {
  TT tt_exact{25};  // 2^25 direct-mapped entries (512 MB) for fully-resolved values
  TT tt_h{22};      // depth-tagged table for horizon-limited searches
  Budget b;

  // Alpha-beta negamax. Precondition: the mover (cur) has NO immediate winning move and
  // n < CELLS - both are enforced by the callers (root wrappers / move filtering; the
  // recursion preserves them: moves that would hand a win are filtered out below).
  int negamax(uint64_t cur, uint64_t mask, int n, int alpha, int beta, int horizon) {
    ++b.nodes;
    if (b.stop()) return 0;
    if (n == CELLS - 1) return 0;  // one stone left, mover cannot win (precondition):
                                   // playing it fills the board => draw
    const bool full = horizon >= CELLS - n;
    const uint64_t key = full ? (cur + mask + 1)
                              : ((cur + mask + 1) | ((uint64_t)(unsigned)horizon << 49));
    TT& tbl = full ? tt_exact : tt_h;
    const size_t ti = tbl.home(key);
    GAI_SOLVER_PREFETCH(&tbl.slot[ti]);  // overlap the TT fetch with the shortcut work
    const uint64_t poss = possible(mask);
    const uint64_t opp = cur ^ mask;
    const uint64_t opp_win = winning(opp, mask);
    const uint64_t forced = poss & opp_win;
    uint64_t moves;
    if (forced) {
      if (forced & (forced - 1)) return score_lose_fast(n);  // double threat: proven loss
      moves = forced & ~(opp_win >> 1);                       // must block the threat
    } else {
      moves = poss & ~(opp_win >> 1);                         // never play under a win square
    }
    if (moves == 0) return score_lose_fast(n);                // every move loses: proven
    // mate-distance window clamps (we cannot win faster than ply n+3 nor lose faster
    // than ply n+4 given a non-losing move exists)
    const int max_s = (CELLS - 1 - n) / 2;
    if (beta > max_s) { beta = max_s; if (alpha >= beta) return beta; }
    const int min_s = -(CELLS - 2 - n) / 2;
    if (alpha < min_s) { alpha = min_s; if (alpha >= beta) return alpha; }
    // transposition table: exact table when the horizon covers the rest of the game
    int32_t ts; uint8_t tf; int8_t tcol = -1;
    if (tbl.probe_at(ti, key, ts, tf, tcol)) {
      if (tf == TT::kExact) return ts;
      if (tf == TT::kLower) {           // true value >= ts
        if (ts >= beta) return ts;
        if (ts > alpha) { alpha = ts; if (alpha >= beta) return ts; }
      } else {                          // kUpper: true value <= ts
        if (ts <= alpha) return ts;
        if (ts < beta) { beta = ts; if (alpha >= beta) return ts; }
      }
    }
    const int a0 = alpha, b0 = beta;
    int best = -1000; int8_t bc = -1;
    // child ordering: best column seen on any earlier probe of this node first (huge win
    // across the null-window binary-search passes, which revisit the same tree), then the
    // static center-out order
    int ord[W];
    for (int i = 0; i < W; i++) ord[i] = ORDER[i];
    if (tcol >= 0 && tcol < W && (moves & COL(tcol))) {
      for (int i = 1; i < W; i++)
        if (ord[i] == tcol) { ord[i] = ord[0]; ord[0] = tcol; break; }
    }
    for (int i = 0; i < W; i++) {
      const int c = ord[i];
      const uint64_t mv = moves & COL(c);
      if (!mv) continue;
      int v = 0;
      if (horizon > 1) v = -negamax(opp, mask | mv, n + 1, -beta, -alpha, horizon - 1);
      if (b.aborted) return 0;
      if (v > best) { best = v; bc = (int8_t)c; }
      if (v > alpha) { alpha = v; if (alpha >= beta) break; }
    }
    if (!b.aborted)
      tbl.store_at(ti, key, (int32_t)best,
                   best <= a0 ? TT::kUpper : (best >= b0 ? TT::kLower : TT::kExact), bc);
    return best;
  }

  // Evaluate a position whose mover MAY have an immediate win (root children and other
  // unfiltered probes; the internal recursion never needs this - see negamax's filtering).
  // Terminal / immediate-win information is exact regardless of the horizon.
  int eval_child(uint64_t cur, uint64_t mask, int n, int alpha, int beta, int horizon) {
    if (n >= CELLS) return 0;  // board full: draw
    if (winning(cur, mask) & possible(mask)) return score_win_now(n);
    if (horizon <= 0) return 0;  // depth-limited horizon reached: conservative unknown
    return negamax(cur, mask, n, alpha, beta, horizon);
  }

  static int win_col_now(uint64_t cur, uint64_t mask) {
    const uint64_t w = winning(cur, mask) & possible(mask);
    if (!w) return -1;
    for (int i = 0; i < W; i++) if (w & COL(ORDER[i])) return ORDER[i];
    return -1;
  }

  struct RootOut { int v; int col; int depth; bool exact; };

  // Exact root solve: null-window binary search on the score (Pons), then best-move
  // extraction: first move (in `order`) whose null-window child value is exactly -v.
  // Precondition: win_col_now(cur, mask) < 0 (no immediate win) and n < CELLS.
  int solve_exact(uint64_t cur, uint64_t mask, int n, const int* order, int* best_col) {
    int lo = score_lose_fast(n), hi = -score_lose_fast(n);  // [-(42-n)/2, (42-n)/2]
    while (lo < hi) {
      int med = lo + (hi - lo) / 2;
      if (med <= 0 && lo / 2 < med) med = lo / 2;
      else if (med >= 0 && hi / 2 > med) med = hi / 2;
      const int r = negamax(cur, mask, n, med, med + 1, CELLS);
      if (b.aborted) return 0;
      if (r <= med) hi = r; else lo = r;
    }
    const uint64_t poss = possible(mask), opp = cur ^ mask;
    for (int i = 0; i < W; i++) {
      const int c = order[i];
      const uint64_t mv = poss & COL(c);
      if (!mv) continue;
      const int r = eval_child(opp, mask | mv, n + 1, -lo, -lo + 1, CELLS);
      if (b.aborted) break;
      if (r == -lo) { if (best_col) *best_col = c; return lo; }
    }
    if (best_col)  // fallback (abort path only): first playable column in `order`
      for (int i = 0; i < W; i++)
        if (poss & COL(order[i])) { *best_col = order[i]; break; }
    return lo;
  }

  // Budgeted root: iterative deepening d = 1..max_depth over root children in `order`.
  // `order` holds up to W column ids; entries outside [0, W) (e.g. the -1 padding AbAgent
  // leaves in unused slots) are skipped. Deterministic tie handling: first child in
  // `order` achieving the best value wins.
  RootOut solve_id(uint64_t cur, uint64_t mask, int n, const int* order, int max_depth) {
    const uint64_t poss = possible(mask), opp = cur ^ mask;
    const int wcol = win_col_now(cur, mask);
    if (wcol >= 0) return {score_win_now(n), wcol, 1, true};
    int best_v = 0, best_c = -1, done = 0; bool exact = false;
    for (int d = 1; d <= max_depth && !b.aborted; d++) {
      int alpha = -1000, bv = -1000, bc = -1;
      for (int i = 0; i < W; i++) {
        const int c = order[i];
        if (c < 0 || c >= W) continue;
        const uint64_t mv = poss & COL(c);
        if (!mv) continue;
        const int v = -eval_child(opp, mask | mv, n + 1, -1000, -alpha, d - 1);
        if (b.aborted) break;
        if (v > bv) { bv = v; bc = c; }
        if (v > alpha) alpha = v;
      }
      if (b.aborted) {
        // Partial depth. Only a proven win (bv > 0) is a sound exact value: a single
        // fully-resolved winning child proves the position, but "all *searched* children
        // lose" (bv < 0) does NOT - an unsearched child may still save the game - and a
        // partial 0 is just a hint. So: adopt proven wins as exact, adopt 0 only as a
        // plain heuristic move when nothing completed yet, never adopt a partial loss.
        if (bc >= 0 && (bv > 0 || (bv == 0 && done == 0))) {
          best_v = bv; best_c = bc; done = d;
          exact = bv > 0;
        }
        break;
      }
      best_v = bv; best_c = bc; done = d;
      if (bv != 0 || d >= CELLS - n) { exact = true; break; }
    }
    if (best_c < 0)
      for (int i = 0; i < W; i++) {
        const int c = order[i];
        if (c >= 0 && c < W && (poss & COL(c))) { best_c = c; break; }
      }
    return {best_v, best_c, done, exact};
  }
};

inline C4Solver& c4_solver() {  // persistent TTs across calls (one instance per thread)
  static thread_local C4Solver inst;
  return inst;
}

// ---------------------------------------------------------------------------
// Generic solver: any gai::GameLike game, engine values +1/0/-1, ascending move order
// ---------------------------------------------------------------------------
template <class G> struct GenericSolver {
  using State = typename G::State;
  TT tt_exact{20};  // fully-resolved values (horizon covers the whole remaining game)
  TT tt_h{18};      // depth-tagged table for horizon-limited searches
  Budget b;

  // Precondition: !G::is_terminal(s) and horizon >= 1.
  int negamax(const State& s, int horizon, int alpha, int beta) {
    ++b.nodes;
    if (b.stop()) return 0;
    const bool full = horizon >= G::kMaxGameLength;
    const uint64_t key = full ? G::hash(s)
                              : (G::hash(s) ^ ((uint64_t)(unsigned)horizon * 0x9E3779B97F4A7C15ULL));
    TT& tbl = full ? tt_exact : tt_h;
    const size_t ti = tbl.home(key);
    int32_t ts; uint8_t tf; int8_t tcol = -1;
    if (tbl.probe_at(ti, key, ts, tf, tcol)) {
      if (tf == TT::kExact) return ts;
      if (tf == TT::kLower) {
        if (ts >= beta) return ts;
        if (ts > alpha) { alpha = ts; if (alpha >= beta) return ts; }
      } else {
        if (ts <= alpha) return ts;
        if (ts < beta) { beta = ts; if (alpha >= beta) return ts; }
      }
    }
    Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv);
    const int a0 = alpha, b0 = beta;
    int best = -2; int8_t bm = -1;
    for (int i = 0; i < n; i++) {
      State t = s; G::apply(t, mv[i]);
      int v;
      if (G::is_terminal(t)) v = -(int)G::outcome(t, G::current_player(t));  // exact
      else if (horizon <= 1) v = 0;                                          // conservative hint
      else v = -negamax(t, horizon - 1, -beta, -alpha);
      if (b.aborted) return 0;
      if (v > best) { best = v; bm = (int8_t)mv[i]; }
      if (v > alpha) { alpha = v; if (alpha >= beta) break; }
    }
    if (!b.aborted)
      tbl.store_at(ti, key, (int32_t)best,
                   best <= a0 ? TT::kUpper : (best >= b0 ? TT::kLower : TT::kExact), bm);
    return best;
  }

  struct RootOut { int v; int mv; int depth; bool exact; };

  // Single-pass exact root scan (no budgets): children at full horizon, rising alpha,
  // deterministic: first move in `order` with the best value wins.
  RootOut solve_root_full(const State& s, const Move* order, int n_order) {
    int alpha = -2, bv = -2, bm = -1;
    for (int i = 0; i < n_order; i++) {
      State t = s; G::apply(t, order[i]);
      int v;
      if (G::is_terminal(t)) v = -(int)G::outcome(t, G::current_player(t));
      else v = -negamax(t, G::kMaxGameLength, -2, -alpha);
      if (b.aborted) return {0, n_order > 0 ? (int)order[0] : -1, 0, false};
      if (v > bv) { bv = v; bm = (int)order[i]; }
      if (v > alpha) alpha = v;
    }
    return {bv, bm, G::kMaxGameLength, true};
  }

  // Budgeted root: iterative deepening, same adoption rules as C4Solver::solve_id.
  RootOut solve_root_id(const State& s, const Move* order, int n_order, int max_depth) {
    int best_v = 0, best_m = -1, done = 0; bool exact = false;
    for (int d = 1; d <= max_depth && !b.aborted; d++) {
      int alpha = -2, bv = -2, bm = -1;
      for (int i = 0; i < n_order; i++) {
        State t = s; G::apply(t, order[i]);
        int v;
        if (G::is_terminal(t)) v = -(int)G::outcome(t, G::current_player(t));
        else if (d <= 1) v = 0;
        else v = -negamax(t, d - 1, -2, -alpha);
        if (b.aborted) break;
        if (v > bv) { bv = v; bm = (int)order[i]; }
        if (v > alpha) alpha = v;
      }
      if (b.aborted) {
        // Same soundness rule as C4Solver::solve_id: a partial depth only ever adopts a
        // proven win (bv > 0, exact) or a 0 heuristic move when nothing completed yet.
        if (bm >= 0 && (bv > 0 || (bv == 0 && done == 0))) {
          best_v = bv; best_m = bm; done = d;
          exact = bv > 0;
        }
        break;
      }
      best_v = bv; best_m = bm; done = d;
      if (bv != 0 || d >= G::kMaxGameLength) { exact = true; break; }
    }
    if (best_m < 0 && n_order > 0) best_m = (int)order[0];
    return {best_v, best_m, done, exact};
  }
};

template <class G> GenericSolver<G>& generic_solver() {
  static thread_local GenericSolver<G> inst;
  return inst;
}

}  // namespace solver_detail

// A Connect-4-style game whose State is the Tromp bitboard layout (fast path).
template <class G>
concept Connect4Bitboard = GameLike<G> && (G::kActionCount == 7) && (G::kInputW == 7) &&
    (G::kInputH == 6) && (G::kMaxGameLength == 42) && requires(typename G::State s) {
  { s.cur } -> std::convertible_to<uint64_t>;
  { s.mask } -> std::convertible_to<uint64_t>;
  { s.n } -> std::convertible_to<int>;
};

namespace solver_detail {
inline int c4_sign(int score) { return score > 0 ? 1 : (score < 0 ? -1 : 0); }
inline float c4_p0_value(int score_ptm, int ptm_player) {
  const int sgn = c4_sign(score_ptm);
  return ptm_player == 0 ? (float)sgn : -(float)sgn;
}
}  // namespace solver_detail

// Solve one position. node_budget / time_budget_ms <= 0 mean unlimited. Unlimited runs
// produce the exact game-theoretic value; budgeted runs use iterative deepening and mark
// partial-depth hints with exact=false (non-zero values are always proven).
template <class G>
Solved solve_state(const typename G::State& s, long node_budget = 0, long time_budget_ms = 0) {
  Solved r;
  const auto t0 = std::chrono::steady_clock::now();
  if (G::is_terminal(s)) {
    r.value_from_p0 = G::outcome(s, 0);
    r.best_move = -1; r.exact = true; r.depth_done = 0;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
  }
  if constexpr (Connect4Bitboard<G>) {
    using solver_detail::C4Solver;
    C4Solver& sv = solver_detail::c4_solver();
    sv.b.set(node_budget, time_budget_ms);
    const uint64_t cur = s.cur, mask = s.mask;
    const int n = s.n, p = G::current_player(s);
    if (node_budget <= 0 && time_budget_ms <= 0) {
      int v, bc = -1;
      const int wcol = C4Solver::win_col_now(cur, mask);
      if (wcol >= 0) { v = C4Solver::score_win_now(n); bc = wcol; }
      else v = sv.solve_exact(cur, mask, n, C4Solver::ORDER, &bc);
      r.score_ptm = v; r.exact = true; r.depth_done = C4Solver::CELLS - n;
      r.best_move = bc;
      r.value_from_p0 = solver_detail::c4_p0_value(v, p);
    } else {
      const auto out = sv.solve_id(cur, mask, n, C4Solver::ORDER, C4Solver::CELLS - n);
      r.score_ptm = out.v; r.exact = out.exact; r.depth_done = out.depth;
      r.best_move = out.col;
      r.value_from_p0 = solver_detail::c4_p0_value(out.v, p);
    }
    r.nodes = sv.b.nodes;
    sv.b.set(0, 0);  // never leak budgets into the shared instance
  } else {
    auto& gs = solver_detail::generic_solver<G>();
    gs.b.set(node_budget, time_budget_ms);
    Move mv[G::kMaxMoves];
    int n = G::legal_moves(s, mv);  // ascending order: deterministic tie handling
    const int p = G::current_player(s);
    typename solver_detail::GenericSolver<G>::RootOut out{0, -1, 0, false};
    if (node_budget <= 0 && time_budget_ms <= 0) out = gs.solve_root_full(s, mv, n);
    else out = gs.solve_root_id(s, mv, n, G::kMaxGameLength);
    r.score_ptm = out.v;
    r.value_from_p0 = p == 0 ? (float)out.v : -(float)out.v;
    r.best_move = out.mv; r.exact = out.exact; r.depth_done = out.depth;
    r.nodes = gs.b.nodes;
    gs.b.set(0, 0);
  }
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

// Batch solve. best_out[i] = best move for the player to move in positions[i];
// value_out[i] = +1/0/-1 from the perspective of the player to move (engine convention).
// Returns the number of positions processed.
template <class G>
int solve_(std::vector<typename G::State>& positions, int* best_out, float* value_out) {
  int k = 0;
  for (size_t i = 0; i < positions.size(); i++) {
    const Solved r = solve_state<G>(positions[i]);
    best_out[i] = r.best_move;
    const float v = r.value_from_p0;
    value_out[i] = G::current_player(positions[i]) == 0 ? v : -v;
    k++;
  }
  return k;
}

}  // namespace gai
