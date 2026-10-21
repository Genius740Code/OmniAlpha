// solve_oracle: exact-solver self-test, oracle-labelled evaluation data, and agent sanity.
//
//   solve_oracle --game connect4 --solve-initial
//       Self-test anchor: solving the initial position must report a first-player win
//       (+1) with the center column as the winning first move (known result). Prints the
//       solving stats (nodes visited, seconds).
//
//   solve_oracle --game connect4 --positions K --out labels.bin
//       [--seed S] [--plies-min L] [--plies-max M] [--per-position-nodes B]
//       [--verify-children]
//       Generates K distinct random reachable positions by seeded random self-play from
//       random openings, solves each EXACTLY, and writes binary labels. Prints the number
//       of solved successes and total seconds.
//
//   solve_oracle --game connect4 --crosscheck N
//       Differential check: solves N late-game positions exactly and compares against a
//       naive brute-force negamax on games/*/reference_game.hpp (deliberately different
//       implementation), positions at [--crosscheck-min P] plies or deeper.
//
//   solve_oracle --game connect4 --ab-selftest
//       Sanity checks for the AbAgent arena agent: immediate win, forced block, center
//       preference, determinism, heuristic fallback.
//
// Label file format (little-endian host order):
//   header: char magic[8] = "GAISLBL1", u32 record_count, u32 record_size=16, u32 reserved=0
//   record: u64 G::hash(state), f32 value_from_p0 (+1/0/-1), i16 best_move, u8 plies, u8 exact
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>
#include "engine/evaluation/ab_agent.hpp"
#include "engine/evaluation/solver.hpp"
#include "tools/common.hpp"
using namespace gai;

static void usage() {
  std::fprintf(stderr,
               "usage: solve_oracle --game NAME [--solve-initial]\n"
               "                    [--positions K --out FILE [--seed S] [--plies-min L]\n"
               "                       [--plies-max M | --max-plies-from M]\n"
               "                       [--per-position-nodes B] [--verify-children]]\n"
               "                    [--crosscheck N [--crosscheck-min P] [--crosscheck-max P]\n"
               "                       [--seed S] [--brute-cap B]]\n"
               "                    [--ab-selftest]\n");
}

// ---------------------------------------------------------------------------
// Position generation: seeded random self-play, position sampled at a random ply.
// ---------------------------------------------------------------------------
template <class G> struct GenPos {
  typename G::State s;
  std::vector<Move> hist;  // moves leading to s (plies = hist.size())
};

template <class G>
static std::vector<GenPos<G>> generate_positions(long count, long plies_min, long plies_max,
                                                Rng& rng) {
  std::vector<GenPos<G>> out;
  std::unordered_set<Hash> seen;
  long attempts = 0;
  while ((long)out.size() < count && attempts < count * 200 + 100) {
    attempts++;
    auto s = G::initial();
    std::vector<Move> hist;
    while (!G::is_terminal(s)) {
      Move mv[G::kMaxMoves];
      const int n = G::legal_moves(s, mv);
      const Move m = mv[rng.below((uint32_t)n)];
      G::apply(s, m);
      hist.push_back(m);
    }
    const long hi = std::min(plies_max, (long)hist.size() - 1);
    if (hi < plies_min) continue;  // game ended too early; try another random game
    const long t = plies_min + (long)rng.below((uint32_t)(hi - plies_min + 1));
    auto ps = G::initial();
    for (long i = 0; i < t; i++) G::apply(ps, hist[i]);
    if (!seen.insert(G::hash(ps)).second) continue;  // distinct positions only
    hist.resize((size_t)t);
    out.push_back({ps, std::move(hist)});
  }
  return out;
}

// ---------------------------------------------------------------------------
// --solve-initial
// ---------------------------------------------------------------------------
template <class G> static int solve_initial(const Args& a) {
  const auto s = G::initial();
  const Solved r = solve_state<G>(s);
  std::printf("[solve-initial] game=%s value_from_p0=%+.0f best_move=%d exact=%s nodes=%ld "
              "seconds=%.3f score_ptm=%d depth_done=%d\n",
              a.str("game").c_str(), r.value_from_p0, r.best_move, r.exact ? "true" : "false",
              r.nodes, r.seconds, r.score_ptm, r.depth_done);
  bool pass = r.exact;
  if (a.str("game") == "connect4") {
    // Known result: Connect-4 is a first-player win, fastest forced win via the center
    // column; under win-distance scoring the initial score is +1 (win completed with the
    // 41st stone of 42).
    pass = r.exact && r.value_from_p0 == 1.f && r.best_move == 3 && r.score_ptm == 1;
    if (pass)
      std::printf("SELF-TEST PASSED: first player wins (value +1), best first move = center "
                  "column %d; matches the known game-theoretic result\n",
                  r.best_move);
    else
      std::printf("SELF-TEST FAILED: expected value_from_p0=+1, best_move=3, score=+1\n");
  } else {
    std::printf("info: no hard-coded anchor for game '%s'; exact=%s, value_from_p0=%+.0f\n",
                a.str("game").c_str(), r.exact ? "true" : "false", r.value_from_p0);
  }
  return pass ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --positions K --out FILE
// ---------------------------------------------------------------------------
struct LabelRecord {  // 16 bytes, no padding; little-endian host order
  uint64_t hash;
  float value_from_p0;
  int16_t best_move;
  uint8_t plies;
  uint8_t exact;
};
static_assert(sizeof(LabelRecord) == 16);

template <class G> static int generate_labels(const Args& a) {
  const long K = a.num("positions", 0);
  const std::string out_path = a.str("out");
  if (out_path.empty()) {
    std::fprintf(stderr, "error: --positions requires --out FILE\n");
    return 2;
  }
  const long plies_min = std::max(1L, a.num("plies-min", 8));
  const long plies_max = std::max(plies_min, a.num("plies-max", a.num("max-plies-from", 34)));
  const long ppn = a.num("per-position-nodes", 0);
  Rng rng((uint64_t)a.num("seed", 12345));

  const auto t0 = std::chrono::steady_clock::now();
  auto positions = generate_positions<G>(K, plies_min, plies_max, rng);
  if ((long)positions.size() < K) {
    std::fprintf(stderr, "error: could only generate %zu/%ld distinct positions "
                         "(widen [--plies-min/--plies-max])\n", positions.size(), K);
    return 1;
  }
  const double gen_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("[labels] game=%s positions=%ld seed=%ld plies=[%ld,%ld] per-position-nodes=%ld "
              "(generation %.2fs)\n",
              a.str("game").c_str(), K, a.num("seed", 12345), plies_min, plies_max, ppn, gen_secs);

  long solved = 0, skipped = 0, nodes_total = 0;
  int dist[3] = {0, 0, 0};  // p0 win / draw / p1 win
  std::vector<LabelRecord> records;
  std::vector<size_t> solved_idx;  // records[i] <-> positions[solved_idx[i]]
  records.reserve(K);
  const long report = std::max(1L, K / 8);
  for (long i = 0; i < K; i++) {
    const Solved r = ppn > 0 ? solve_state<G>(positions[i].s, ppn) : solve_state<G>(positions[i].s);
    nodes_total += r.nodes;
    if (!r.exact) {  // never write an unproven label
      skipped++;
      continue;
    }
    const LabelRecord rec{G::hash(positions[i].s), r.value_from_p0, (int16_t)r.best_move,
                          (uint8_t)positions[i].hist.size(), 1};
    records.push_back(rec);
    solved_idx.push_back((size_t)i);
    dist[r.value_from_p0 > 0 ? 0 : (r.value_from_p0 < 0 ? 2 : 1)]++;
    solved++;
    if (solved % report == 0 || i + 1 == K)
      std::printf("  [%4ld/%ld solved] last %.3fs, nodes=%ld value=%+.0f best=%d plies=%u\n",
                  solved, K, r.seconds, r.nodes, r.value_from_p0, r.best_move,
                  (unsigned)positions[i].hist.size());
  }
  const double solve_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - gen_secs;

  FILE* f = std::fopen(out_path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "error: cannot open '%s' for writing\n", out_path.c_str());
    return 1;
  }
  const char magic[8] = {'G', 'A', 'I', 'S', 'L', 'B', 'L', '1'};
  const uint32_t cnt = (uint32_t)records.size(), rec_size = (uint32_t)sizeof(LabelRecord), res = 0;
  std::fwrite(magic, 1, 8, f);
  std::fwrite(&cnt, 4, 1, f);
  std::fwrite(&rec_size, 4, 1, f);
  std::fwrite(&res, 4, 1, f);
  for (const LabelRecord& r : records) std::fwrite(&r, sizeof(r), 1, f);
  std::fclose(f);

  for (size_t i = 0; i < records.size() && i < 3; i++)
    std::printf("[record %zu] hash=%016llx plies=%u value_from_p0=%+.0f best_move=%d\n", i,
                (unsigned long long)records[i].hash, records[i].plies, records[i].value_from_p0,
                records[i].best_move);
  std::printf("[value distribution] p0_win=%d draw=%d p1_win=%d\n", dist[0], dist[1], dist[2]);
  std::printf("[labels] wrote %zu records (magic \"GAISLBL1\", 20-byte header, 16-byte records) -> %s\n",
              records.size(), out_path.c_str());

  if (a.has("verify-children")) {
    long bad = 0, checked = 0;
    for (size_t i = 0; i < records.size(); i++) {
      const size_t pi = solved_idx[i];
      auto c = positions[pi].s;
      G::apply(c, (Move)records[i].best_move);
      const Solved rc = solve_state<G>(c);
      const int p = G::current_player(positions[pi].s);
      const float parent_mover = p == 0 ? records[i].value_from_p0 : -records[i].value_from_p0;
      const int pc = G::current_player(c);
      const float child_mover = pc == 0 ? rc.value_from_p0 : -rc.value_from_p0;
      checked++;
      if (!rc.exact || -child_mover != parent_mover) {
        bad++;
        std::printf("[verify] MISMATCH at record %zu: parent(mover)=%+.0f child(mover)=%+.0f\n",
                    i, parent_mover, child_mover);
      }
    }
    std::printf("[verify-children] re-solved %ld best-move children, negamax relation "
                "v(parent) == -v(child) holds: %ld/%ld\n", checked, checked - bad, checked);
    if (bad) return 1;
  }

  {  // exercise the batch API solve_() and cross-compare against the per-position results
    const size_t nsmoke = std::min<size_t>(records.size(), 3);
    if (nsmoke > 0) {
      std::vector<typename G::State> smoke;
      for (size_t i = 0; i < nsmoke; i++) smoke.push_back(positions[solved_idx[i]].s);
      std::vector<int> bmove(nsmoke, -1);
      std::vector<float> bval(nsmoke, 0.f);
      const int processed = solve_<G>(smoke, bmove.data(), bval.data());
      long bad = 0;
      for (size_t i = 0; i < nsmoke; i++) {
        const int p = G::current_player(smoke[i]);
        const float want_mover = p == 0 ? records[i].value_from_p0 : -records[i].value_from_p0;
        if (bval[i] != want_mover || bmove[i] != records[i].best_move) bad++;
      }
      std::printf("[batch API] solve_() processed %d positions, mismatches vs records: %ld\n",
                  processed, bad);
      if (bad) return 1;
    }
  }

  std::printf("[stats] nodes_total=%ld avg_nodes=%ld avg_solve_ms=%.1f skipped_unproven=%ld\n",
              nodes_total, solved > 0 ? nodes_total / solved : 0,
              solved > 0 ? solve_secs * 1000.0 / solved : 0.0, skipped);
  std::printf("solved %ld/%ld positions in %.2fs (%.1f ms/position)\n", solved, K,
              solve_secs + gen_secs, solved > 0 ? solve_secs * 1000.0 / solved : 0.0);
  return solved == K ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --crosscheck N: exact solver vs naive brute force on the reference implementation
// ---------------------------------------------------------------------------
template <class R, int kBuf> static int brute(const typename R::State& s, long& nodes, long cap) {
  if (R::is_terminal(s)) return (int)R::outcome(s, R::current_player(s));
  if (++nodes > cap) return 0;
  Move mv[kBuf];
  const int n = R::legal_moves(s, mv);
  int best = -2;
  for (int i = 0; i < n; i++) {
    auto t = s;
    R::apply(t, mv[i]);
    const int v = -brute<R, kBuf>(t, nodes, cap);
    if (v > best) best = v;
  }
  return best;
}

template <class T> static int crosscheck(const Args& a) {
  using G = typename T::Game;
  using R = typename T::Ref;
  const long N = std::max(1L, a.num("crosscheck", 40));
  const long plies_min = std::max(0L, a.num("crosscheck-min", 36));
  const long plies_max = a.num("crosscheck-max", 40);
  const long cap = a.num("brute-cap", 4000000);
  Rng rng((uint64_t)a.num("seed", 12345));
  const auto positions = generate_positions<G>(N, plies_min, plies_max, rng);
  if ((long)positions.size() < N) {
    std::fprintf(stderr, "error: could only generate %zu/%ld positions\n", positions.size(), N);
    return 1;
  }
  std::printf("[crosscheck] game=%s positions=%ld plies>=%ld brute_cap=%ld\n",
              a.str("game").c_str(), N, plies_min, cap);
  if constexpr (std::is_void_v<R>) {
    std::printf("[crosscheck] no reference_game.hpp -> skipped\n");
    return 0;
  } else {
    long bad = 0, checked = 0, overrun = 0;
    for (const auto& gp : positions) {
      const Solved r = solve_state<G>(gp.s);
      const int p = G::current_player(gp.s);
      const int mover = p == 0 ? (r.value_from_p0 > 0 ? 1 : (r.value_from_p0 < 0 ? -1 : 0))
                              : (r.value_from_p0 > 0 ? -1 : (r.value_from_p0 < 0 ? 1 : 0));
      auto rs = R::initial();
      for (Move m : gp.hist) R::apply(rs, m);  // same position, reference implementation
      long nodes = 0;
      const int bv = brute<R, G::kMaxMoves + 8>(rs, nodes, cap);
      if (nodes > cap) { overrun++; continue; }
      checked++;
      if (bv != mover) {
        bad++;
        std::printf("[crosscheck] MISMATCH at ply %zu: solver=%+d brute=%+d\n",
                    gp.hist.size(), mover, bv);
      }
    }
    std::printf("[crosscheck] exact solver vs naive reference brute force: %ld/%ld positions "
                "agree, %ld mismatches, %ld brute overruns\n",
                checked - bad, checked, bad, overrun);
    return bad ? 1 : 0;
  }
}

// ---------------------------------------------------------------------------
// --ab-selftest
// ---------------------------------------------------------------------------
template <class G> static int ab_selftest(const Args& a) {
  AbAgent<G> agent;
  int fails = 0;
  auto check = [&](const char* what, int got, int want) {
    const bool ok = got == want;
    if (!ok) fails++;
    std::printf("  %-38s expected=%d got=%d %s\n", what, want, got, ok ? "ok" : "FAIL");
  };
  std::printf("[ab-selftest] game=%s\n", a.str("game").c_str());
  if (a.str("game") == "connect4") {
    {  // immediate win: p0 owns row 0 of columns 0,1,2 AND is to move; finish at column 3
      auto s = G::initial();
      for (int m : {0, 6, 1, 6, 2, 0}) G::apply(s, (Move)m);
      check("depth-6 agent takes immediate win", agent.select_move(s, 6, 2000), 3);
      if (agent.last.score_ptm != (43 - 6) / 2 || !agent.last.exact) {
        fails++;
        std::printf("  %-38s score=%d exact=%d FAIL\n", "win value is exact", agent.last.score_ptm,
                    (int)agent.last.exact);
      }
      check("depth-0 heuristic takes immediate win", agent.select_move(s, 0, 0), 3);
    }
    {  // immediate win on the EDGE column: p0 has a vertical three at col 6 (and p1 also
      // threatens at col 3 - winning must outrank blocking, center bias must not win)
      auto s = G::initial();
      for (int m : {6, 0, 6, 1, 6, 2}) G::apply(s, (Move)m);
      check("edge-column immediate win beats block", agent.select_move(s, 6, 2000), 6);
      check("depth-0 heuristic, same position", agent.select_move(s, 0, 0), 6);
    }
    {  // forced block on the EDGE column: p1 has a vertical three at col 0; p0 has no win
      auto s = G::initial();
      for (int m : {1, 0, 2, 0, 4, 0}) G::apply(s, (Move)m);
      check("depth-6 agent blocks edge threat", agent.select_move(s, 6, 2000), 0);
      check("depth-2 agent blocks edge threat", agent.select_move(s, 2, 2000), 0);
    }
    {  // forced block: p1 owns row 0 of columns 0,1,2 (threat at col 3); p0 has no win
      auto s = G::initial();
      for (int m : {6, 0, 6, 1, 5, 2}) G::apply(s, (Move)m);
      check("depth-6 agent blocks single threat", agent.select_move(s, 6, 2000), 3);
      check("depth-2 agent blocks single threat", agent.select_move(s, 2, 2000), 3);
    }
    {  // opening: no proof within depth 8 -> deterministic center-bias tie-break
      auto s = G::initial();
      const int m1 = agent.select_move(s, 8, 1000);
      check("depth-8 opening move is center", m1, 3);
      const int m2 = agent.select_move(s, 8, 1000);
      check("reproducible at same depth/time", m2, m1);
    }
  } else {
    // generic game: legality + determinism only (heuristics are connect4-specific)
    auto s = G::initial();
    const int m = agent.select_move(s, 4, 1000);
    Move mv[G::kMaxMoves];
    const int n = G::legal_moves(s, mv);
    bool legal = false;
    for (int i = 0; i < n; i++) legal |= (Move)m == mv[i];
    if (m < 0 || !legal) {
      fails++;
      std::printf("  agent returned illegal move %d (legal:", m);
      for (int i = 0; i < n; i++) std::printf(" %d", (int)mv[i]);
      std::printf(") FAIL\n");
    } else std::printf("  %-38s move=%d ok\n", "legal move from initial", m);
    const int m2 = agent.select_move(s, 4, 1000);
    check("reproducible at same depth/time", m2, m);
  }
  std::printf(fails ? "AB-AGENT SELF-TEST FAILED (%d)\n" : "AB-AGENT SELF-TEST PASSED\n", fails);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------

template <class T> static int run(const Args& a) {
  using G = typename T::Game;
  if (a.has("solve-initial")) return solve_initial<G>(a);
  if (a.has("ab-selftest")) return ab_selftest<G>(a);
  if (a.has("crosscheck")) return crosscheck<T>(a);
  if (a.num("positions", 0) > 0) return generate_labels<G>(a);
  usage();
  return 2;
}

int main(int argc, char** argv) {
  Args a(argc, argv);
  if (!a.has("game")) {
    usage();
    return 2;
  }
  return dispatch_game(a.str("game"), [&](auto tag) { return run<decltype(tag)>(a); });
}
