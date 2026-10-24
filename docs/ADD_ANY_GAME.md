# ADD ANY GAME — fast, effective, in 30–60 minutes

Goal of this repo: **strongest bot per wall-clock minute**. This file is the fast path from
"no game" to "trained bot I can play against", for anything from chess to Monopoly.
It replaces/absorbs `docs/ADDING_A_GAME.md`. The engine contract is documented in
`engine/core/api.hpp` — you never touch `engine/`, only `games/<name>/`.

---

## 0. The 10-second version

```bash
python3 tools/new_game.py <name> --from board      # scaffold: take-away | board | dice
# edit games/<name>/game.hpp (the ONLY logic file) + reference_game.hpp
cmake --build build -j                              # once per session
./build/validate_game --game <name>                 # MUST print VALIDATION PASSED
./build/play --game <name> --p0 human --p1 mcts --sims 400   # bot plays before any training!
# train (see §6), then play the improving net live:
./build-lt/play --game <name> --p0 human --p1 mcts --model models/latest.ts --reload --tc 2+15
```

Why this is fast:
- **CMake auto-registers** every `games/<name>/game.hpp` (namespace `game_<name>`); no engine edits.
- All tools (`validate_game`, `benchmark_game`, `selfplay`, `evaluate`, `play`, `solve_oracle`)
  and the whole Python training pipeline work for ANY conforming game automatically.
- The differential oracle (`reference_game.hpp`) catches the classic hand-implementer bugs
  (move numbering, player toggling, side effects, zero-sum violations) in minutes.
- `./build/play ... --p1 mcts` gives a *rollout-evaluated* MCTS bot immediately — useful to sanity
  check playability long before the first training run.

Templates (choose the closest family; copy the closest idea from `games/demo_*`):

| template | family | what you change | typical time |
|---|---|---|---|
| `take-away` | pure logic (tic-tac-toe on 1D, take-away, nim, chicken/dare) | ~10 small functions | 15–30 min |
| `board` | grid board games: connect-K, gomoku, othello, chess-lite | constants + mover-set + side effects | 30–60 min |
| `dice` | chance + race + capture: monopolyish, ludo, sorry, backgammon-lite | squares/bonuses/cards/chance-from-chain | 45–90 min |

Working, validated examples of each family: `games/template` (take-away),
`games/demo_board` (8x8 connect-4-style bitboard, 8 symmetries), `games/demo_dice`
(hash-chain dice race with capture). Read them beside this file.

---

## 1. The mandatory edit loop (never skips validation)

1. Implement `game.hpp` — every hook has a `TODO`/explanation comment in the template.
2. Implement `reference_game.hpp` — same rules, **deliberately different representation**
   (grid of chars where the main game is a bitboard; history vector where the main game is a
   counter). Its whole purpose is to disagree when you are wrong.
3. `./build/validate_game --game <name>` until `VALIDATION PASSED` (it random-playouts both
   implementations in lockstep and diffs legal moves / outcomes / undo / symmetries / zero-sum).
4. `./build/benchmark_game --game <name>` — for a 1-hour jam you need **legal_moves > 5e6/s and
   apply+hash > 5e6/s** (connect4 is ~8e6). If below, switch to bitboards/incremental updates.
5. Optional but 30 seconds: run once with sanitizers
   (`cmake -B build-asan -DGAI_SANITIZE="address;undefined" -DCMAKE_BUILD_TYPE=Debug && ...`).

Validation red flags & fixes (validator prints each):
- legal moves outside `[0, kActionCount)` → renumber moves (`move id == policy index`, always).
- random games exceed `kMaxGameLength` → raise it (it is a hard validator bound).
- `outcome()` not zero-sum → must satisfy `outcome(s,0) == -outcome(s,1)`.
- "legal moves differ from reference at ply K" → race: your oracle and your fast game disagree;
  add a 10-line debug main (include both headers, print both rolls/legal lists) — the bug is
  nearly always in one side effect order or an off-by-one.
- infinite/very long random games: ensure progression (some rule that shrinks distance or bounds).

---

## 2. The API you must implement (contract, from `engine/core/api.hpp`)

| hook | meaning | gotchas |
|---|---|---|
| `kActionCount` | policy size; **move value == policy index** | big values need big nets; see §3 |
| `kMaxMoves` | upper bound of legal moves in any position | `EvalResult` stores priors in this width |
| `kMaxGameLength` | hard ply bound | tight = faster training sampling |
| `initial()` | fresh state | keep State tiny & trivially copyable (MCTS copies per sim) |
| `legal_moves(s, out)` | ascending move ids, return count | WRONG ORDER = silently wrong priors |
| `apply(s, m)` | mutate in place, flip `current_player` | illegal m = UB; all randomness must come from state (§4.2) |
| `is_terminal(s)` | end position | make apply detect wins incrementally when possible |
| `outcome(s, p)` | +1 win / 0 draw / −1 loss, zero-sum | only meaningful in terminal states |
| `current_player(s)` | 0 or 1 | exactly two players (engine constraint) |
| `hash(s)` | eval/transposition cache key | every piece of state that affects legal moves must change it |
| `encode(s, out)` | NN input planes, **from the mover's view** | layout `[plane][h][w]`; zero what you don't set |
| `transform_move`, `symmetry_input` | symmetries (`kNumSymmetries > 1`) | required hooks even when identity (only 1) |
| optional `undo(s, m)` | validated when present | enables solver probes; skip when awkward |

Engine limitations to respect: two players, ids 0/1, strictly alternating from player 0,
deterministic rules, zero-sum outcomes. Single-player, 3+ player, or hidden-info games need
engine changes first — out of scope for a faster/stronger game day (see §7).

---

## 3. Designing the game TO TRAIN WELL (this is where strength comes from)

Two games can share rules yet differ 10× in Elo-per-hour. Choose implementations with training
value, not just "what compiles":

1. **Small policy vector wins.** `kActionCount` = the NN output width. Prefer move ids that
   describe the actor's choice compactly (column, square, die-face, card-slot). If your natural
   action space is huge (chess ≈4680 like-size encodings), you are choosing a harder training
   problem: more data, more batches, slower convergence — fine for long-term training, risky for
   a 1-hour jam. A 1-hour jam loves `kActionCount ≤ ~64`.
2. **Symmetries are free data augmentation.** A board game with geometric symmetry gains ×N
   effective samples from `kNumSymmetries` (connect4: 2; square grids: 8 — validated in
   `demo_board`). Do NOT burn jam time on exotic symmetries; do take mirror/transpose when free.
3. **Cheap `encode`.** Encode is called per leaf (batched, but still millions of times). Table:
   - planes "my pieces / their pieces" (mover's view) = the basics;
   - add "empty", repetition counters, clocks, phase flags only if the rules really need memory.
   - `benchmark_game` shows `encode_per_s`; keep it same magnitude as `apply_per_s`.
4. **Bitboards beat arrays.** State copy cost × sims-per-second is your entire search budget.
   Demo examples: connect4 24-byte state at 8e6 apply/s; demo_board 24 B at 6e6/s. If your game
   naturally needs a big state (deck arrays, piece lists), prefer per-hash-derived determinism
   (see §4.2) or compact indices — never carry `std::vector`/pointers inside `State`.
5. **Tight `kMaxGameLength` and small branching.** The engine already exploits: win/block-in-1
   shortcuts, forced playouts, can't-be-overtaken early stop, solver proofs. Small branching and
   bounded length make these fire MUCH more often (demo_dice branch ≈3.5 → strong shortcut rate).
6. **Deterministic rules = MCTS-able rules.** MCTS here has no chance nodes. If your game is
   stochastic by nature, do NOT emit RNG inside `apply` (it breaks tree reuse AND the validator).

      ### §4.2 The chance trick (dice/cards/monopoly — REQUIRED pattern)
   Derive all chance from a **public hash-chain in State**:
   ```cpp
   // roll for this turn:
   int roll = 1 + (gai::mix64(chain * K1) >> 25) % 6;
   // every apply advances it, so rolls differ between games but stay deterministic:
   chain = gai::mix64(chain + K2 * (uint64_t)(move + 1));
   ```
   Costs: one static opening (first roll identical every game — same as chess book) and the bot
   can, in principle, memorize a long fixed sequence. Both are irrelevant at jam scale and keep
   self-play/validator/tree-reuse intact. The network still reads the roll as an input plane
   (`demo_dice::encode` plane 2), so it learns to plan under known dice.
   "Both players may pick their own die" is NOT a substitute — it degenerates to always-max.
   Player-owned randomness (real RNG in `apply()`/`legal_moves`) breaks determinism — never.
7. **Reference implementation = cheap insurance.** The 15 minutes writing the slow obvious version
   is ALWAYS worth it: the validator's differential test caught a planted diagonal bug; side
   effects (captures, slides, card draws) are where fast implementations go wrong.

Effort ladder (pick deliberately, jam clock is the constraint):
- **Level 0 (15 min):** take-away template clone — trains and plays immediately; use when the new
  game ideas are the point and the rules are simple.
- **Level 1 (30–60 min):** board template clone for any "pieces on a grid" game.
- **Level 2 (1–2 h):** dice template clone for chance games; then swap squares/bonuses for YOUR
  board pieces (money squares, card decks = another `mix64` chain, auction phases = fold phases
  into move ids).
- **Level 3 (multi-day):** chess/othello-class; needs careful legal-move enumeration, undo, maybe
  solver support. Do NOT attempt inside a 1-hour jam; see §7 for what to defer.

---

## 4. Game families, concretely

### 4.1 How to spec a **chess-class game** fast
Do NOT write your rules from raw imagination in one pass; split:
1. Board + piece encoding (bitboards recommended, bit index = square).
2. Move generation per piece type with **pin-free pseudo-legality first**; legality via
   "does my king fall into attack after this move" using the oracle as ground truth.
3. Special moves: castling/en passant/promotion — encode promotion as extra move ids.
4. Territory state in State (castling rights, halfmove clock, en-passant square) — 1-2 bytes each;
   all must be part of `hash`.
5. Extra engine notes: repetition draws exist → include a repetition counter plane in `encode`;
   a 4682-wide policy vector is standard.
For the 1-hour format, a **reduced chess** (no castling, no en passant, 3-fold → 2-fold) trains
dramatically faster and still plays recognizably.

### 4.2 How to spec a **Monopoly-class game** fast
Monopoly is >2 players + hidden rolls + purchases + bankruptcy. Within this engine do:
1. Two-player heads-up (or "2-player per table" if you must).
2. Chance = hash-chain dice (§4.2 pattern above); card decks = second chain (cars = extra players
   not possible — buyable cards act as owned state flags instead).
3. Purchases as extra move ids: ("buy street", "don't buy") in a multi-phase turn encoded by the
   move id itself: e.g. move id = event*10 + option. `apply` walks the phases by unfolding state
   (phase counter in State, "whose micro-turn" is part of current_player semantics — free).
4. Values (rents/prices) live in constants; encode money as a plane (scale/normalise!).
5. Bankruptcy = is_terminal with outcome vs the *other* player; never model nim-global positions.

---

## 5. Playing: humans vs the bot

```bash
./play --game <name> --p0 human --p1 mcts --sims 400             # no training needed
./play --game <name> --p0 mcts --p1 mcts --tc 15+1               # bot vs bot with clocks
./evaluate --game <name> --games 40 --sims-a 800 --sims-b 100    # arena + Elo/CI (also tests baseline)
```
Game authors: make `to_string` print **both** the board and a hint of what move ids mean —
players type move ids (`play` now lists every legal id each turn). The NN does not use
`to_string`; it is pure human UX.

---

## 6. Training, then playing against the IMPROVING bot while it trains

The full training pipeline (`selfplay → replay → train → TorchScript export → arena promotion`)
is game-agnostic. Two rhythm options:

### A) Long-term training
```bash
PYTHONPATH=python python3 -m gai.orchestrator --config configs/long_term.yaml --game <name>
```

### B) Jam mode / competition (train ≈50 min, game 20 min, TC 2+15 — see docs/HANDOFF.md "Jam plan")
```bash
PYTHONPATH=python python3 -m gai.orchestrator \
  --config configs/competition_60m.yaml --game <name> --minutes 50
```
While that runs (checkpoints landing every ~5 min, `models/latest.ts` re-exported ATOMICALLY every
iteration), play against the live bot:
```bash
./build-lt/play --game <name> --p0 human --p1 mcts \
  --model models/latest.ts --reload --tc 2+15 \
  --solve-probe 1 --cache 65536        # optional strength extras, see below
```
`--reload` re-loads the model file whenever its mtime changes **and the candidate model
actually loads** (a half-written/racing export can never turn the bot uniform — the previous
evaluator is kept until the new one is proven good). `--solve-probe 1` runs the exact solver at
the root each move (budget `--solve-probe-ms`, default 250) and plays any PROVEN win instantly,
saving those sims for contested positions. `--cache [CAP]` wraps the NN in a hash-keyed eval
cache; for NN serving the leaf batch defaults to 32 (`--batch` overrides). `hybrid:`, `cache:` and
`nn-strict:` serving specs all support `--reload`. Resign judge, MCTS-Solver and speculation
shortcuts make serving moves fast; the bot reserves proper thinking for contested midgame
positions (adaptive search does this automatically).
Final strongest checkpoint: endgame Bradley-Terry round-robin picks the champion automatically
(`docs/HANDOFF.md` §Artifacts plan); it is exported as `models/champion.pt(.ts)`.

---

## 7. Known engine limitations to plan around (not fixable in games/<name> alone)

- exactly 2 players, alternating, deterministic, zero-sum (see §2).
- no chance nodes in MCTS → use the §4.2 hash-chain trick.
- no transposition table → long repetitive games (chess endgames) waste sims; keep hash clean.
- "apply illegal move = UB" — validator + `play` both only emit legal moves.
- hidden-information games are out until an engine change (none planned); approximate with
  open-handed or deterministic "everyone can see the deck" variants (fine for jams, less optimal
  for human-tournament rules).

## 8. After validation — the 2-minute strength checklist

- [ ] `./build/benchmark_game` ≥ jam targets (apply+legal ≥5e6/s; encode ≥1e6/s)
- [ ] `./build/play` human game feels right; move ids printed and readable
- [ ] `./build/evaluate --games 40` shows rollout-bot win rate sane and no crashes
- [ ] `python3 -m gai.orchestrator --config configs/competition_60m.yaml --game <name> --minutes 1 --dry-run`
      (see orchestrator flags) for a 1-minute timeline sanity check before the real run
- [ ] an oracle/solver probe if your game is small enough: `./build/solve_oracle --game <name> --solve-initial`
      (solved games give exact value labels for free training data)
