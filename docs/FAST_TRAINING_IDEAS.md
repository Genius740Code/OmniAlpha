# Fast-Training Ideas for `gai` — what other projects do, and what to build

Goal: **strongest bot in ~60 minutes of wall-clock training** (and fast long-term improvement). Every idea below is
judged by one question: *does it raise Elo per wall-clock minute?*

## How to read this document

Every idea carries an **evidence tag**, because some of this comes from papers I read in this session and some is
classic knowledge or my own proposal:

| Tag | Meaning |
|---|---|
| **[SRC]** | Backed by a source I retrieved this session (URL in §10). Numbers quoted are *the source's claims*, not measured here. |
| **[SRC-2nd]** | Only seen via a secondary source (forum post, review, blog). Treat as a lead to verify. |
| **[CLASSIC]** | Standard technique from game-search literature, from general knowledge (not re-verified this session). |
| **[MINE]** | My proposal/synthesis. Plausible, unproven. |

**Nothing here has been measured in this repo yet.** There is no trained network, so no idea has been shown to
improve Elo for `gai`. The backlog in §8 is ordered by expected payoff, but the real order must come from A/B tests (§9).

**Not investigated this session** (no sources retrieved, so I make no claims about their code): Leela Zero internals,
Minigo, OpenSpiel's AlphaZero, mctx source code, EfficientZero details, the original AlphaZero/AlphaGo Zero papers.
Section 7 says what to check in each.

---

## 1. Your core idea: "stop searching once the answer is already clear"

> "No need to search 1000 other moves if it already found M1 or M2 — it's unlikely it lost M1."

This is a real and well-known family of ideas. The key is **how certain you are**, because the cost of being wrong
differs hugely. Three certainty tiers:

| Tier | Meaning | Safe to stop? | Technique |
|---|---|---|---|
| **A. Proven** | The move is a *proven* forced win (or all alternatives proven worse) | **Yes, always** | MCTS-Solver / endgame solver / 1–2 ply tactical check |
| **B. Mathematically unreachable** | The runner-up can't catch the leader in the remaining budget | **Yes** (for the move choice) | Lc0-style "smart pruning" / "can't be overtaken" |
| **C. Statistically dominated** | Leader is ahead with high confidence | **Mostly** — tune with Elo tests | Racing/elimination, Sequential Halving, confidence bounds |
| **D. Heuristic "looks obvious"** | Net policy is peaked, value is stable | **Risky** | Easy-move rules; must be validated by arena Elo |

### 1.1 Tier B — "can't be overtaken" **[SRC-2nd]**
Lc0's *smart pruning* stops the search when no other move can overtake the current best given the remaining budget,
and a forum explanation notes the extra cycles after that point are wasted *if you only care about the best move*
(they are not wasted if you want an accurate ranking of several moves).
- **Already in `gai`:** `adaptive_search` stops when `best_visits − second_visits > target_sims − sims_done`.
- **Improvements to make [MINE]:**
  - Use the *real* remaining budget (`max_sims` or time-based estimate via measured sims/s), not only `target_sims`.
  - Apply the same test to **time**: `remaining_ms × sims_per_ms` as the remaining-sims estimate.
  - Allow a **"safety factor"** (Lc0 exposes a tunable factor; values above 1 prune earlier, 0 disables it).
  - Account for virtual loss / in-flight batch leaves when computing the gap.

### 1.2 Tier A — proven results (MCTS-Solver) **[SRC]**
MCTS-Solver (Winands, Björnsson, Saito 2008) marks terminal nodes as proven win/loss and **propagates the proof up the
tree minimax-style**, so a node is "solved" without more sampling. In their Lines-of-Action experiments it beat plain
MCTS with a 65% score.
Design for `gai` (fits the current `Node` struct) **[MINE, standard algorithm]**:
```
Node gains:  int8_t proven;   // 0 unknown, +1 proven WIN for mover, -1 proven LOSS for mover, 2 proven DRAW
Terminal leaf:  proven = outcome from the view of Node::mover  (win/loss/draw).
Backup (after normal stats update), at parent P choosing among children C:
   - if ANY child is proven WIN for P's chooser  -> P is proven WIN for chooser; stop descending here
   - if ALL children are proven LOSS for chooser -> P is proven LOSS
   - if all children are proven and the best is DRAW -> P is proven DRAW
Selection: never descend into a proven-LOSS child unless all are; prefer proven-WIN child immediately.
Root: if a root child is proven WIN -> STOP SEARCH NOW, play it (Tier A).  If every non-loss move is proven -> stop.
```
- Caveat: with draws the simple win/loss solver needs the "score-bounded" extension (also noted in Winands' survey).
- **Honest caution [SRC-2nd]:** one blog's independent measurement found a solver improved self-play *mirror* matches
  but did not help against a different opponent in their drawn-ish game; benefits concentrate in tactical/short-horizon
  endgames. Measure it for *your* game before keeping it.

### 1.3 Tier A (cheap) — instant tactical shortcuts **[MINE]**
Before any search, spend O(branching) work on rules that need no game knowledge beyond the API:
1. **Single legal move → play it, do no search, emit no training sample.** (Saves a whole search, and a forced move teaches the net nothing about preference.)
2. **Win-in-1:** for each legal move `apply → is_terminal && outcome==win` → play it.
3. **Block-in-1:** if the opponent has a win-in-1 after a null-like pass (not always available) — instead detect by checking, for each of our moves, whether the opponent still has a winning reply; if exactly one move leaves none, play it.
   (Fully generic: 2-ply exhaustive check, cost ≈ b².)
These are the cases MCTS wastes hundreds of sims on early in training. They also make early random-ish nets look sane faster.
**Training note:** shortcut moves should still be recorded for *value* targets but, KataGo-style, usually not as policy targets (see §2.1).

### 1.4 Tier C — racing / elimination at the root **[CLASSIC + SRC]**
"M1 and M2 are strong, so the other 30 moves are unlikely to matter" is exactly **Sequential Halving**:
- Start with the top-`m` moves (by policy, Gumbel-perturbed), give each equal sims, **drop the worse half**, repeat; all remaining budget concentrates on the survivors.
- This is the core of Gumbel AlphaZero (§2.3), which reports good policy improvement even with very few simulations.
- A simpler stand-in **[MINE]**: UCB-style *racing* — after every K sims, drop root moves whose upper confidence bound on Q is below the best move's lower bound; never revisit them. Cheap to implement in `adaptive_search`.
- Risk: dropping a move the net underrates. Mitigate with Gumbel sampling (stochastic top-m) or a minimum sim count per candidate.

### 1.5 Tier D — "easy move" heuristics **[CLASSIC, matches Lc0 'insta-move' protection]**
Stop early when: policy entropy low **and** best move stable over the last half of the search **and** root value stable.
`gai` already tracks stability/contested ratio; add `value_delta` and require agreement between the **net's prior top-1** and **search top-1**. Always protect against "insta-moves" with a minimum sim count (Lc0 passes an nps estimate to its stopper for this purpose **[SRC-2nd]**).

### 1.6 Important: early stopping hurts *training data* differently than *play*
- **During play:** stopping early is nearly free strength — spend time where it matters.
- **During self-play training:** the policy target is the visit distribution. A truncated or one-sided search gives a *sharper, less informative* target, and training on it can collapse exploration. KataGo's answer is **playout cap randomization** (§2.1): most moves get a cheap search (used only for value/game progression), a random fraction get a full search (used for policy targets).
- Rule of thumb **[MINE]**: *shortcuts and early stops are for choosing moves; only full-budget searches (or Gumbel's improved policy) are used as policy targets.*

---

## 2. Ideas from the strongest open projects (what they do that `gai` doesn't yet)

### 2.1 KataGo — "Accelerating Self-Play Learning in Go" **[SRC]**
The paper claims a **50× reduction in computation** versus comparable methods (e.g. ELF OpenGo-class runs), and says
much of the speedup comes from non-domain-specific changes that should transfer to other games. Its general techniques:

| Idea | What it does | Applies to `gai`? |
|---|---|---|
| **Playout cap randomization** | Randomly vary search depth per move: most moves get a cheap search, some get a full one; only full searches produce policy targets. Balances *data quantity* (more positions/hour) against *target quality*. | **Yes — top priority.** Needs a `full_search` flag in each `Sample` + masked policy loss in Python. |
| **Forced playouts + policy target pruning** | Force a minimum number of playouts for each root child (≈ `sqrt(k · P(c) · ΣN)`), then *prune* those forced visits out of the policy target so training isn't taught to like noise moves. Decouples exploration from the training target. | **Yes** (medium). Pure MCTS-side change + target post-processing. |
| **Auxiliary policy targets** | Also predict the player's *future* action(s) (revived from supervised Go) to speed learning. | Yes if the game has long horizons; cheap extra head. |
| **Global pooling in the net** | Net architecture tweak for board-global context. | Game-dependent (large boards). |
| **Ownership/score targets, input features** | Go-specific extra targets. | Only analogues (e.g. "territory", "material") — game-specific. |
| **First-play urgency (FPU)** | Unvisited-child value derived from explored policy mass; paper uses a constant c_FPU = 0.2. | `gai` has a simpler FPU (parent Q − 0.25). Tune. |
| **Stochastic weight averaging; gating** | Averaged weights; a candidate must beat the current net in a ~200-game test to replace it. | SWA: cheap Python add. Gating: but see §6 (gating costs wall-clock). |

### 2.2 Go-Exploit — "Targeted Search Control" **[SRC]**
Instead of always starting self-play from the initial position, **start trajectories from an archive of "states of interest"**
(states visited in earlier self-play). Reported benefits: better exploration of the game tree, a value function that
generalizes better, and **shorter trajectories → more independent value targets per second**. Tested on **Connect Four and 9×9 Go**
(i.e. directly relevant to our example game) with greater sample efficiency than AlphaZero, and it was shown to combine
with KataGo's other improvements. A 2026 follow-up (*Regret-guided search control*) claims further gains by prioritizing
high-regret states instead of sampling uniformly, reporting average gains of 77 and 89 Elo over AlphaZero and Go-Exploit
respectively (snippet-level; verify before relying on the numbers).
- **For `gai` [MINE]:** add an `Archive<G>` of states (ring buffer by hash) + `--start-from-archive p` to `selfplay`; mix, e.g., 50% standard starts / 50% archive starts. Needs only `G::State` copy — game-agnostic.

### 2.3 Gumbel AlphaZero / Gumbel MuZero (DeepMind) **[SRC]**
Standard AlphaZero can fail to improve the policy if the root doesn't visit all actions. The Gumbel variants sample root
actions **without replacement (Gumbel-Top-k)**, allocate budget with **Sequential Halving**, and select/update using a
principled *improved policy* instead of heuristics. The paper shows strong results with few simulations (experiments with
as few as 2 sims) and a policy-improvement guarantee given correct action values. LightZero's benchmark review also reports Gumbel MuZero
beating standard MuZero when simulations are limited **[SRC-2nd]**.
- **Why it matters for 60-minute training:** simulation count is the budget. Fewer sims per move ⇒ many more games/hour, *and* the target is still an improvement.
- **For `gai`:** implement `GumbelRoot` in `engine/mcts/` as a selectable root policy (keep PUCT below the root); emit the *improved policy* as the training target. Compare vs PUCT+Dirichlet by Elo-per-minute at equal wall-clock (§9).
- Existing implementation to study: DeepMind's `mctx` (JAX) and a Rust crate `treant-gumbel` that advertises Gumbel-Top-k, Sequential Halving and PUCT below the root **[SRC-2nd]** (check licenses before reusing).

### 2.4 ELF OpenGo / AlphaZero reproductions **[SRC / SRC-2nd]**
- A comprehensive open reimplementation with ablations; reported superhuman strength after massive compute (thousands of GPUs) — a reminder that *vanilla AZ is very expensive*; the improvements above are what make small budgets viable.
- **Game resignation** during self-play (stop hopeless games early) is reported to focus the net on opening/middlegame faster **[SRC-2nd]**; the slides also note adaptive resign thresholds have delays. For `gai`: add `--resign-threshold` with a **calibration step** that keeps a % of games un-resigned to measure false-resign rate **[MINE]**.

### 2.5 Leela Chess Zero **[SRC-2nd]**
- *Smart pruning* (see §1.1) is its time-saving early stop.
- Not investigated: its search-parameter ecosystem (cpuct schedule, FPU modes, policy temperature, "moves-left" head). Worth reading its docs for tuning ranges.

### 2.6 LightZero (OpenDILab, NeurIPS 2023) **[SRC / SRC-2nd]**
- A unified PyTorch toolkit with AlphaZero, MuZero, EfficientZero, Sampled MuZero, Gumbel MuZero, Stochastic MuZero, and later variants; its MCTS exists in **both Python and C++**; benchmarks include **TicTacToe, Connect4 and Gomoku**.
- A secondary review reports: **AlphaZero is notably more sample-efficient than MuZero on board games when a perfect simulator exists** — i.e. for our setting (rules known) *don't* use MuZero; keep AlphaZero-style + Gumbel. EfficientZero's tricks (learned model, value-prefix, self-supervised consistency) target the *no-simulator* case and are low priority here.
- Use as a **reference for hyperparameters and baselines** on Connect4; check license first.

---

## 3. Search-side speed (more positions per second)

| Idea | Tag | Notes for `gai` | Status |
|---|---|---|---|
| Bitboards / incremental updates | CLASSIC | Per-game; validator + reference diff makes this safe | Connect-4 done |
| **NN evaluation cache keyed by `G::hash`** | CLASSIC | Many positions repeat (transpositions, tree reuse). Cache (hash → priors, value). Needs a symmetry-canonical hash to also catch mirrored duplicates. Most valuable once NN is slow. | TODO |
| Transposition table in the tree (DAG) | CLASSIC | Shares stats between paths; careful with backup (cycles/double counting). Try the NN cache first (simpler, safer). | TODO |
| Batched leaf evaluation + virtual loss | CLASSIC/SRC | Core of GPU efficiency | Done (single tree); **cross-game batching TODO** |
| Many games per process feeding one GPU queue | CLASSIC | Biggest GPU-utilization win; hide CPU search latency behind inference | TODO |
| Tree reuse | CLASSIC | Already done; some report it helps self-play more than cross-opponent play **[SRC-2nd]** — measure | Done |
| Avoid `State` copies deep in tree | MINE | Use `apply/undo` during descent on one scratch state; `undo` hook exists | Optional |
| FP16/AMP, channels-last, CUDA graphs, TensorRT/int8 | CLASSIC | Only after profiling says the net is the bottleneck | TODO |
| Smaller net early, grow later | MINE/CLASSIC | Early positions are low quality; a small fast net generates more data per minute. Switch/distill to larger net mid-run (racing hook). | TODO |

## 4. Training-side speed (more learning per position)

| Idea | Tag | Notes |
|---|---|---|
| **Symmetry augmentation** | CLASSIC | `transform_move` / `symmetry_input` already validated for Connect-4; need export of permutation tables to Python. ~2× data for mirror-symmetric games, free. |
| Value target mixing (game outcome z vs search value q) | CLASSIC | Lower-variance targets early; tune λ |
| Auxiliary targets (future policy, moves-left, score) | SRC (KataGo) | Extra heads; cheap, often faster early learning |
| Replay window sizing & sample-reuse ratio | CLASSIC | Too-small window ⇒ overfit; too-large ⇒ stale. Make "train steps per new position" a tracked, tunable ratio |
| Overlap self-play and training (async) | CLASSIC | Keep GPU busy: train while workers generate; sync weights every N minutes |
| LR warmup + cosine/step schedule, AdamW/SGD, large batch | CLASSIC | Tune per game in config racing |
| Stochastic weight averaging | SRC (KataGo) | Cheap stability gain |
| Opening diversity (random opening plies, archive starts) | SRC (Go-Exploit) / CLASSIC | Avoid training on near-identical games |
| Resignation / early game termination with calibration | SRC-2nd | Saves self-play time on lopsided games |
| Curriculum: start from endgames/small positions | CLASSIC/MINE | Endgame positions have precise values, a solver can label them exactly (§5) |

## 5. Hybrids: classical search & solvers (your alpha-beta / transposition-table / bitboard instinct)

AlphaZero-style MCTS is not always the best use of 60 minutes. For some games a classical engine wins:

- **Alpha-beta + iterative deepening + transposition table + move ordering (killers/history) + bitboards [CLASSIC]** is a very strong baseline for small/tactical games with cheap, decent evaluation. A well-written one can beat a net trained for an hour.
- **Practical uses inside `gai` [MINE]:**
  1. **Baseline opponent / referee** for Elo: always know "are we beating a solid alpha-beta?" instead of only "beating our old self".
  2. **Endgame solver:** when ≤ K plies/pieces remain, solve exactly (alpha-beta or retrograde analysis) → *exact* value + best move. Use it as a search shortcut (Tier A) and as **noise-free training targets**.
  3. **Proof-number search / PN-MCTS** for forced-win-heavy games: Winands' group reports PN-MCTS outperforming MCTS in most tested domains (tactical games, not Gomoku) **[SRC]**.
- **Decision rule:** if the game's evaluation is easy to hand-write *and* tactics dominate → build the alpha-beta baseline first and measure; if evaluation is hard/positional → NN+MCTS. Run both in the arena.
- For Connect-4 specifically a perfect solver is publicly known to exist (general knowledge, not verified here) — using one as an *oracle referee* would give an absolute accuracy measure for the trained bot.

## 6. Evaluation speed (don't burn the hour measuring)

- **Gating costs wall-clock.** KataGo gates new nets with a ~200-game test **[SRC]** — fine on a cluster, expensive in 60 minutes. Options **[MINE]**: skip gating in competition mode and promote the latest net, but still run a *cheap* sanity arena (e.g. 20–40 games, low sims) and keep the previous checkpoint as fallback; use **SPRT (sequential test)** to stop an arena as soon as the result is decisive.
- **Paired openings** (play each opening from both colors) cut variance — currently TODO in `evaluate`.
- Track **Elo vs wall-clock** only from a fixed reference opponent (random, MCTS-rollout-N, alpha-beta) so curves are comparable across experiments; arena-vs-previous-self hides regressions.

## 7. Repo-by-repo: what to check next (so the next agent doesn't repeat my gaps)

| Project | What I actually verified | What to inspect next |
|---|---|---|
| **KataGo** | Paper techniques & claims (§2.1) | `cpp/search/` for forced playouts & target pruning; self-play config for playout-cap values; actual license file |
| **Gumbel AZ / mctx** | Paper abstract/claims; a Rust crate's description | `mctx` Gumbel root & `qtransform` (improved-policy value transform) — port the math to C++ |
| **Go-Exploit** | Abstract/conclusion, Connect-4 results | Archive structures (the paper studies several); sampling schedule |
| **Leela Chess Zero** | Smart-pruning behavior via forum | Search params in `src/mcts/params.cc`; time manager / stoppers |
| **LightZero** | README-level facts (algorithms list, C++ MCTS, board-game benchmarks) | Connect4/Gomoku AlphaZero configs & hyperparameters; C++ MCTS batch design |
| **ELF OpenGo** | Abstract; secondary notes on resignation | Resign-threshold calibration code; distributed self-play architecture |
| **OpenSpiel, Minigo, Leela Zero, EfficientZero, original AZ/AGZ papers** | **Nothing — not investigated** | Everything. Verify licenses before copying any code |

## 8. Prioritized backlog (expected payoff for a 60-minute run)

Ordering is by *my expectation*, not measurement; re-order after §9 experiments.

**Tier 0 — prerequisite (nothing else matters without it)**
- [ ] NN evaluator + GPU batching + cross-game batching (HANDOFF steps 1–3)

**Tier 1 — cheap, likely big**
- [ ] Single-legal-move skip + win-in-1 / block-in-1 shortcuts (§1.3)
- [ ] "Can't be overtaken" using real remaining budget/time (§1.1)
- [ ] **Playout cap randomization** + `full_search` flag in samples + masked policy loss (§2.1)
- [ ] Symmetry augmentation (export permutation tables) (§4)
- [ ] Reference-opponent Elo curve (random / rollout-N / alpha-beta) for honest progress tracking (§6)
- [ ] NN eval cache by hash (§3)
- [ ] Overlap self-play and training (§4)

**Tier 2 — medium effort, likely good**
- [ ] MCTS-Solver proof propagation + immediate stop on proven root win (§1.2)
- [ ] **Gumbel root + Sequential Halving** and improved-policy targets (§2.3)
- [ ] Forced playouts + policy target pruning (§2.1)
- [ ] Go-Exploit-style state archive starts (§2.2)
- [ ] Resign threshold with calibration (§2.4)
- [ ] Root racing/elimination via confidence bounds (§1.4)
- [ ] Alpha-beta baseline engine + endgame solver (§5)
- [ ] SPRT arenas + paired openings (§6)
- [ ] Small-net-first schedule inside configuration racing (§3)

**Tier 3 — only if profiling/Elo tests justify**
- [ ] DAG transposition table; auxiliary targets; SWA; CUDA graphs/TensorRT/int8; PN-MCTS; regret-guided search control; MuZero/EfficientZero (low priority: rules are known)

## 9. How to decide what to keep (experiment protocol)

1. **One reference ladder:** fixed opponents (random, MCTS-rollout-100, alpha-beta if available, a frozen earlier net). Report Elo vs each.
2. **A/B at equal wall-clock:** run baseline and variant for the same minutes on the same hardware; compare Elo-vs-minutes curves, not final steps or sims/s.
3. **Seeds/variance:** ≥3 seeds for decisions that matter; self-play runs are noisy.
4. **Per-change checklist:** throughput (positions/s), data quality (policy-target entropy, value loss), Elo at 10/20/30/60 min.
5. **Kill rules:** drop an idea if it doesn't beat baseline at 60 min even if it wins early (or vice-versa for long-term mode).
6. Keep every shortcut behind a config flag so ablations are one-line changes.
7. **Correctness guard for shortcuts:** add arena tests asserting a shortcut never changes the chosen move in *proven* positions (e.g., solver-labelled test suites) — and that early-stop never picks a move the full search ranks as a proven loss.

## 10. Sources (retrieved this session; check dates and licenses yourself)

- KataGo paper, *Accelerating Self-Play Learning in Go* (Wu): https://arxiv.org/abs/1902.10565 · repo: https://github.com/lightvector/KataGo
- *Policy improvement by planning with Gumbel* (Danihelka et al., ICLR 2022): https://iclr.cc/virtual/2022/poster/6418 · (via mlanthology) https://mlanthology.org/iclr/2022/danihelka2022iclr-policy
- Rust Gumbel search crate (secondary): https://docs.rs/treant-gumbel
- Lc0 smart pruning discussions (forums, secondary): https://forums.nextchessmove.com/t/lczero-doesnt-search-for-30-seconds/966 · https://talkchess.com/viewtopic.php?p=872593
- *Monte-Carlo Tree Search Solver* (Winands, Björnsson, Saito 2008): https://staff.ru.is/yngvi/pdf/WinandsBS08.pdf · survey: https://dke.maastrichtuniversity.nl/m.winands/documents/Encyclopedia_MCTS.pdf
- Proof-number MCTS (Winands et al.): https://arxiv.org/abs/2303.09449
- Independent solver measurement blog (anecdotal, secondary): https://brainwagon.org/blog/2026_07_15_the_proof_that_didnt_travel
- ELF OpenGo: https://arxiv.org/abs/1902.04522 · slides: https://folk.idi.ntnu.no/odderik/RAI-2019/presentations/alphazero_fair.pdf · forum summary: https://talkchess.com/viewtopic.php?p=789746
- LightZero: https://github.com/opendilab/lightzero · NeurIPS 2023 paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/765043fe026f7d704c96cec027f13843-Abstract.html · review (secondary): https://liner.com/review/lightzero-a-unified-benchmark-for-monte-carlo-tree-search-in
- Go-Exploit, *Targeted Search Control in AlphaZero*: https://arxiv.org/abs/2302.12359 · Regret-guided search control (2026): https://arxiv.org/abs/2602.20809

## 11. What this means for the next agent (short version)

1. Build the NN evaluator and batching first; everything else is meaningless without a trained net.
2. Add the cheap early-stop/shortcut layer (§1.1, §1.3) and playout cap randomization — highest value per line of code.
3. Then Gumbel root + Sequential Halving, MCTS-Solver, archive starts, resign-with-calibration.
4. Build a **reference ladder** (random / rollout / alpha-beta) so you can *see* whether any of this helps. If it doesn't show up in Elo-per-minute, delete it.
