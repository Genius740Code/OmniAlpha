// PUCT MCTS. Single tree, split select/expand/backup so batched + asynchronous variants share code.
// Value convention: Node::w is summed from the view of Node::mover (the player who chose the edge INTO the node),
// so a parent can read children's Q directly as "good for me".
#pragma once
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include "engine/core/api.hpp"
#include "engine/core/rng.hpp"
#include "engine/inference/evaluator.hpp"

namespace gai {
struct MctsConfig {
  float c_puct = 1.5f;
  float fpu_reduction = 0.25f;     // unvisited child Q = parent Q - fpu_reduction
  float dirichlet_alpha = 0.0f;    // 0 disables root noise
  float dirichlet_eps = 0.25f;
  size_t max_nodes = 4'000'000;    // compact() on advance_root when exceeded
  bool forced_playouts = false;    // KataGo-style: guarantee min visits per root child
  float forced_k = 2.0f;            // n_forced(c) = sqrt(k * prior(c) * total_root_visits)
  // Real Gumbel root + Sequential Halving (Danihelka et al. ICLR 2022), see gumbel_root.hpp.
  // Driven explicitly via run_gumbel_root(); the old prior-noise approximation was A/B killed.
  bool gumbel = false;              // selfplay uses run_gumbel_root for full-budget searches
  int gumbel_sims = 32;             // root simulation budget for Gumbel search
  float gumbel_c_visit = 50.f;      // mctx maxvisit_init; inert in spread mode
  float gumbel_c_scale = 0.1f;      // mctx value_scale; inert in spread mode
  bool gumbel_sigma_mctx = false;   // false=spread-matched sigma (default, knobs inert
                                    // by construction); true=mctx (c_visit+maxN)*c_scale*minmax(Q)
  float prior_temp = 1.0f;          // A1: root+tree prior softmax temp; 1.0 = off.
                                    // effective prior = prior^(1/temp), temp>1 flattens
  bool loss_fallthrough = true;     // all-proven-loss node falls through to PUCT
                                    // (max-resistance) instead of leftmost child
};

template <GameLike G> class MCTS {
 public:
  using State = typename G::State;
  struct Node {
    int32_t parent = -1, first_child = -1;
    Move move = -1; int16_t nchild = 0;
    int8_t mover = -1; float prior = 0;
    int8_t proven = 0; // 0 unknown, +1 proven WIN for mover, -1 proven LOSS for mover, 2 proven DRAW
    int32_t n = 0, vl = 0; float w = 0;
  };
  struct Leaf { int node; State state; bool terminal; };

  explicit MCTS(MctsConfig c = {}) : cfg_(c) {}

  void set_root(const State& s) {
    nodes_.clear(); nodes_.emplace_back(); root_ = 0; root_state_ = s; forced_root_ = -1;
  }
  const State& root_state() const { return root_state_; }

  // Move to the child reached by `m`, reusing its subtree when it exists.
  void advance_root(Move m) {
    int c = find_child(root_, m);
    G::apply(root_state_, m);
    if (c < 0) { set_root(root_state_); return; }
    root_ = c; nodes_[c].parent = -1; forced_root_ = -1;
    if (nodes_.size() > cfg_.max_nodes) compact();
  }

  // Root-forcing hook for Sequential Halving: while set, root selection descends
  // into the given root-child index regardless of PUCT. -1 disables.
  void set_forced_root(int idx) { forced_root_ = idx; }
  void clear_forced_root() { forced_root_ = -1; }

  // --- split simulation API --------------------------------------------------------
  Leaf select_leaf() {
    State s = root_state_; int cur = root_; nodes_[cur].vl++;
    while (true) {
      if (G::is_terminal(s)) return {cur, s, true};
      if (nodes_[cur].first_child < 0) return {cur, s, false};
      int best = pick_child(cur, s);
      G::apply(s, nodes_[best].move); cur = best; nodes_[cur].vl++;
    }
  }
  void finish_leaf(const Leaf& l, const EvalResult<G>* r) {
    int P = G::current_player(l.state); float v;
    if (l.terminal) {
      v = G::outcome(l.state, P);
      Node& ln = nodes_[l.node];
      if (ln.first_child < 0) {
        float vm = (ln.mover == P || ln.mover < 0) ? v : -v;
        // Root terminal: proven from player-to-move view; else from mover view.
        if (l.node == root_) ln.proven = (vm > 0 ? (int8_t)1 : vm < 0 ? (int8_t)-1 : (int8_t)2);
        else ln.proven = (vm > 0 ? (int8_t)1 : vm < 0 ? (int8_t)-1 : (int8_t)2);
      }
    } else {
      v = r->value;
      if (nodes_[l.node].first_child < 0) expand(l.node, l.state, *r);  // may already be expanded (batch dup)
    }
    backup(l.node, P, v);
  }
  // Synchronous convenience: n simulations, `batch` leaves per evaluator call (virtual loss spreads leaves).
  void run(int n, Evaluator<G>& ev, int batch = 1) {
    std::vector<Leaf> leaves; std::vector<State> sts; std::vector<EvalResult<G>> res(batch);
    int done = 0;
    while (done < n) {
      leaves.clear(); sts.clear();
      int want = std::min(batch, n - done);
      for (int i = 0; i < want; i++) leaves.push_back(select_leaf());
      for (auto& l : leaves) if (!l.terminal) sts.push_back(l.state);
      if (!sts.empty()) ev.evaluate(sts.data(), res.data(), (int)sts.size());
      int k = 0;
      for (auto& l : leaves) finish_leaf(l, l.terminal ? nullptr : &res[k++]);
      done += want;
    }
  }

  // --- root control --------------------------------------------------------------
  bool root_expanded() const { return nodes_[root_].first_child >= 0; }
  void add_root_noise(Rng& rng) {
    if (cfg_.dirichlet_alpha <= 0 || !root_expanded()) return;
    Node& r = nodes_[root_]; std::vector<float> d(r.nchild); float sum = 0;
    std::gamma_distribution<float> g(cfg_.dirichlet_alpha, 1.0f);
    for (auto& x : d) { x = g(rng); sum += x; }
    for (int i = 0; i < r.nchild; i++)
      nodes_[r.first_child + i].prior = (1 - cfg_.dirichlet_eps) * nodes_[r.first_child + i].prior + cfg_.dirichlet_eps * d[i] / sum;
  }

  // --- statistics ----------------------------------------------------------------
  int total_visits() const { return nodes_[root_].n; }
  size_t node_count() const { return nodes_.size(); }
  int num_root_children() const { return nodes_[root_].nchild; }
  Move child_move(int i) const { return nodes_[nodes_[root_].first_child + i].move; }
  int child_visits(int i) const { return nodes_[nodes_[root_].first_child + i].n; }
  float child_prior(int i) const { return nodes_[nodes_[root_].first_child + i].prior; }
  float child_q(int i) const { const Node& c = nodes_[nodes_[root_].first_child + i]; return c.n ? c.w / c.n : 0.f; }
  void visit_policy(float* out) const {
    for (int a = 0; a < G::kActionCount; a++) out[a] = 0;
    float tot = 0;
    for (int i = 0; i < num_root_children(); i++) tot += child_visits(i);
    if (tot <= 0) return;
    for (int i = 0; i < num_root_children(); i++) out[child_move(i)] = child_visits(i) / tot;
  }
  // KataGo-style improved policy: subtract forced exploration visits from non-best
  // children (prune to 0 when reduced to <=1), renormalize. Best child keeps all visits.
  void improved_policy(float* out) const {
    for (int a = 0; a < G::kActionCount; a++) out[a] = 0;
    int nc = num_root_children();
    if (nc <= 0) return;
    int bi = 0;
    for (int i = 1; i < nc; i++) if (child_visits(i) > child_visits(bi)) bi = i;
    float tot = (float)nodes_[nodes_[root_].first_child + bi].n;
    std::vector<float> kept(nc);
    kept[bi] = (float)child_visits(bi);
    for (int i = 0; i < nc; i++) {
      if (i == bi) continue;
      const Node& c = nodes_[nodes_[root_].first_child + i];
      float nf = 0;
      if (cfg_.forced_playouts) {
        float total = (float)std::max(1, nodes_[root_].n);
        nf = std::sqrt(cfg_.forced_k * prior_eff(c.prior) * total);
      }
      float k = (float)c.n - nf;
      if (k <= 1.0f) k = 0;  // outright prune single-playout children
      kept[i] = k;
      tot += k;
    }
    if (tot <= 0) { visit_policy(out); return; }
    for (int i = 0; i < nc; i++) out[child_move(i)] = kept[i] / tot;
  }
  float root_value() const {  // from the view of the player to move at the root
    if (!root_expanded()) return 0;
    float w = 0; int n = 0;
    for (int i = 0; i < num_root_children(); i++) { const Node& c = nodes_[nodes_[root_].first_child + i]; w += c.w; n += c.n; }
    return n ? w / n : 0.f;
  }
  Move best_move() const {
    int bi = 0; for (int i = 1; i < num_root_children(); i++) if (child_visits(i) > child_visits(bi)) bi = i;
    return child_move(bi);
  }
  // A6 LCB move selection: pick by lower confidence bound on value instead of
  // raw visits, so a barely-visited move with a lucky value isn't chosen.
  // lcb_i = q_i - beta*sqrt(log(N+1)/(n_i+1)), q from chooser's view. beta<=0
  // disables (falls back to best_move). Proven wins short-circuit; proven
  // losses excluded unless all children are lost.
  Move best_move_lcb(float beta) const {
    int nc = num_root_children();
    if (nc <= 0) return -1;
    if (beta <= 0) return best_move();
    int fc = nodes_[root_].first_child;
    for (int i = 0; i < nc; i++) if (nodes_[fc + i].proven == 1) return child_move(i);
    bool all_loss = true;
    for (int i = 0; i < nc; i++) if (nodes_[fc + i].proven != -1) { all_loss = false; break; }
    float tot = 0;
    for (int i = 0; i < nc; i++) tot += (float)child_visits(i);
    float logN = std::log(tot + 1.f);
    int bi = -1; float bs = -1e30f;
    for (int i = 0; i < nc; i++) {
      if (nodes_[fc + i].proven == -1 && !all_loss) continue;
      float q = child_q(i);
      float lcb = q - beta * std::sqrt(logN / ((float)child_visits(i) + 1.f));
      if (bi < 0 || lcb > bs) { bs = lcb; bi = i; }
    }
    return bi < 0 ? best_move() : child_move(bi);
  }
  int8_t root_proven() const { return nodes_[root_].proven; }
  int8_t node_proven(int node) const { return nodes_[node].proven; }
  void top2_visits(int& best, int& second) const {
    best = second = 0;
    for (int i = 0; i < num_root_children(); i++) { int v = child_visits(i); if (v > best) { second = best; best = v; } else if (v > second) second = v; }
  }
  float policy_entropy() const {
    float tot = 0, h = 0; for (int i = 0; i < num_root_children(); i++) tot += child_visits(i);
    if (tot <= 0) return 0;
    for (int i = 0; i < num_root_children(); i++) { float p = child_visits(i) / tot; if (p > 0) h -= p * std::log(p); }
    return h;
  }
  Move sample_move(float temperature, Rng& rng) const {
    if (temperature <= 1e-3f) return best_move();
    std::vector<double> w(num_root_children()); double sum = 0;
    for (int i = 0; i < num_root_children(); i++) { w[i] = std::pow((double)child_visits(i), 1.0 / temperature); sum += w[i]; }
    double r = rng.uniform() * sum;
    for (int i = 0; i < num_root_children(); i++) { r -= w[i]; if (r <= 0) return child_move(i); }
    return child_move(num_root_children() - 1);
  }

 private:
  float prior_eff(float p) const {
    if (cfg_.prior_temp <= 0 || cfg_.prior_temp == 1.0f) return p;
    return std::pow(std::max(p, 0.f), 1.0f / cfg_.prior_temp);
  }
  int find_child(int node, Move m) const {    const Node& n = nodes_[node];
    for (int i = 0; i < n.nchild; i++) if (nodes_[n.first_child + i].move == m) return n.first_child + i;
    return -1;
  }
  float fpu_for(int cur, int chooser) const {
    // Root FPU: no absolute penalty (KataGo uses reduction 0 at root with noise on).
    // Use the root value estimate (mean child Q from the chooser's view), 0 when unvisited.
    // Non-root FPU stays parent Q minus reduction.
    const Node& p = nodes_[cur];
    if (cur == root_) {
      float w = 0; int n = 0;
      for (int i = 0; i < p.nchild; i++) { const Node& c = nodes_[p.first_child + i]; w += c.w; n += c.n; }
      return n ? w / n : 0.f;
    }
    float parent_q = 0;
    if (p.n > 0) parent_q = (p.mover == chooser ? 1.f : -1.f) * p.w / p.n;
    return parent_q - cfg_.fpu_reduction;
  }
  int pick_child(int cur, const State& s) const {
    const Node& p = nodes_[cur]; int chooser = G::current_player(s);
    if (cur == root_ && forced_root_ >= 0 && forced_root_ < p.nchild)
      return p.first_child + forced_root_;
    // Forced playouts at root: PUCT = infinity while child visits < sqrt(k*prior*total).
    if (cfg_.forced_playouts && cur == root_ && p.nchild > 0) {
      float total = (float)std::max(1, p.n);
      for (int i = 0; i < p.nchild; i++) {
        int ci = p.first_child + i; const Node& c = nodes_[ci];
        float need = std::sqrt(cfg_.forced_k * prior_eff(c.prior) * total);
        if ((float)(c.n + c.vl) < need) return ci;
      }
    }
    // Solver steering: instant win, avoid proven loss. Child mover == chooser.
    if (p.nchild > 0) {
      int win = -1, nonloss = -1, nloss = 0;
      for (int i = 0; i < p.nchild; i++) {
        int ci = p.first_child + i; const Node& c = nodes_[ci];
        if (c.proven == 1) { win = ci; break; }
        if (c.proven == -1) nloss++;
        else if (nonloss < 0) nonloss = ci;
      }
      if (win >= 0) return win;
      // All lost: fall through to PUCT over all children (maximum resistance)
      // instead of the leftmost move. Flag preserves the old behavior.
      if (nloss == p.nchild && !cfg_.loss_fallthrough) return p.first_child;
      if (nonloss >= 0 && nloss > 0) {
        // Exclude proven losses from PUCT consideration below by restricting to non-loss.
        // Fall through to PUCT but skip proven-loss children.
        float fpu = fpu_for(cur, chooser);
        float sq = std::sqrt((float)std::max(1, p.n + p.vl - 1));
        int best = nonloss; float bs = -1e30f;
        for (int i = 0; i < p.nchild; i++) {
          int ci = p.first_child + i; const Node& c = nodes_[ci];
          if (c.proven == -1) continue;
          int ne = c.n + c.vl;
          float q = ne ? (c.w - c.vl) / ne : fpu;
          float sc = q + cfg_.c_puct * prior_eff(c.prior) * sq / (1 + ne);
          if (sc > bs) { bs = sc; best = ci; }
        }
        return best;
      }
    }
    float fpu = fpu_for(cur, chooser);
    float sq = std::sqrt((float)std::max(1, p.n + p.vl - 1));
    int best = -1; float bs = -1e30f;
    for (int i = 0; i < p.nchild; i++) {
      int ci = p.first_child + i; const Node& c = nodes_[ci];
      int ne = c.n + c.vl;
      float q = ne ? (c.w - c.vl) / ne : fpu;   // virtual loss counts as a loss for the chooser
      float sc = q + cfg_.c_puct * prior_eff(c.prior) * sq / (1 + ne);
      if (sc > bs) { bs = sc; best = ci; }
    }
    return best;
  }
  void expand(int node, const State& s, const EvalResult<G>& r) {
    Move mv[G::kMaxMoves]; int n = G::legal_moves(s, mv);
    int base = (int)nodes_.size(); nodes_.resize(base + n);
    int mover = G::current_player(s);
    for (int i = 0; i < n; i++) { Node& c = nodes_[base + i]; c.parent = node; c.move = mv[i]; c.prior = r.priors[i]; c.mover = (int8_t)mover; }
    nodes_[node].first_child = base; nodes_[node].nchild = (int16_t)n;
  }
  void backup(int node, int P, float v) {
    for (int x = node; x >= 0; x = nodes_[x].parent) { Node& n = nodes_[x]; n.n++; n.vl--; n.w += (n.mover == P ? v : -v); }
    // MCTS-Solver propagation. proven is from the view of Node::mover:
    // child mover == player to move at parent, so child +1 (win for chooser)
    // means parent -1 (loss for parent mover), and vice versa. Draws pass through.
    for (int x = node; x >= 0; x = nodes_[x].parent) {
      Node& n = nodes_[x];
      if (n.first_child < 0) continue;  // unexpanded: keep current proven (terminal set in finish_leaf)
      bool any_win_for_chooser = false, all_loss_for_chooser = true, all_proven = true;
      for (int i = 0; i < n.nchild; i++) {
        int8_t cp = nodes_[n.first_child + i].proven;
        if (cp == 0) { all_proven = false; all_loss_for_chooser = false; }
        else if (cp == 1) any_win_for_chooser = true, all_loss_for_chooser = false;
        else if (cp == 2) all_loss_for_chooser = false;
        // cp == -1 keeps all_loss_for_chooser true
      }
      if (x == root_) {
        if (any_win_for_chooser) n.proven = 1;
        else if (all_proven && all_loss_for_chooser) n.proven = -1;
        else if (all_proven) n.proven = 2;
        else n.proven = 0;
      } else {
        if (any_win_for_chooser) n.proven = -1;       // chooser wins => mover loses
        else if (all_proven && all_loss_for_chooser) n.proven = 1;  // chooser loses everywhere => mover wins
        else if (all_proven) n.proven = 2;
        else n.proven = 0;
      }
    }
  }
  void compact() {  // copy the live subtree into a fresh pool (BFS)
    std::vector<Node> out; out.reserve(nodes_.size() / 4 + 16);
    std::vector<int> queue{root_}, newidx{0}; Node r = nodes_[root_]; r.parent = -1; out.push_back(r);
    for (size_t qi = 0; qi < queue.size(); qi++) {
      const Node old = nodes_[queue[qi]]; if (old.first_child < 0) { out[newidx[qi]].first_child = -1; continue; }
      int base = (int)out.size(); out[newidx[qi]].first_child = base;
      for (int i = 0; i < old.nchild; i++) { Node c = nodes_[old.first_child + i]; c.parent = newidx[qi]; out.push_back(c); queue.push_back(old.first_child + i); newidx.push_back(base + i); }
    }
    nodes_.swap(out); root_ = 0;
  }
  MctsConfig cfg_; std::vector<Node> nodes_; int root_ = 0; State root_state_{}; int forced_root_ = -1;
};
}  // namespace gai