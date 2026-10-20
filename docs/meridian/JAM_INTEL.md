# Meridian jam intel (JAM_INTEL.md)

Date: 2026-10-20. Site: <https://constellation.blueshrimp.uk> (game: "Constellation",
game name Meridian, rules in `bots/meridian/RULES.md`).

Purpose: everything the Omni effort needs to know from one full data-mining pass over
the site — game/result statistics, a color-balanced bot leaderboard, opening win
rates, score dynamics, ranked improvement ideas, and the jam/tournament facts
(format, deadlines, the resign question, timeout-forfeit rules).

## Data foundation (and a recovery note)

The previous session's scratch (`/tmp/opencode`: unpacked `meridian-games.zip`,
`nn-mining`, `nn-bots.json`, `nn-tournaments.json`) was **gone** — `/tmp/opencode`
was empty. Everything was re-fetched from the site in this pass; no analysis was
carried over. All fetches succeeded (see "Fetch log"). Current scratch lives in
`/tmp/opencode/meridian-games/`:

| file | what it is |
|---|---|
| `meridian-games.zip` + `meridian-games/games.jsonl` | site export, generated 2026-10-20 06:43 UTC, sha256 `f93d25d3…`, 5,099 games |
| `turn-scores.jsonl` | per-turn cumulative scores for all 5,099 games, replayed by the site's own engine |
| `analysis.txt` | the full mining report this document summarises |
| `ep-api-bots.json`, `ep-rankings.json`, `ep-api-tournaments.json` | `/api/bots` (63 public), `/api/rankings` (85 bots), `/api/tournaments` (5) |
| `bot-api.md`, `site.js` | the site's bot API doc and SPA bundle (mined for rules text) |

Replay validation: **all 5,099 games replayed legally through the unmodified
`meridian-engine`, and the replayed final scores match the export exactly (5,099 /
5,099)**. The mining is grounded, not approximated. New tool:
`bots/meridian/omni/src/bin/scores.rs` (replay dumper), run as

```sh
cd bots/meridian/omni && cargo build --release --bin scores
./target/release/scores --games /tmp/opencode/meridian-games/meridian-games/games.jsonl \
  --out /tmp/opencode/meridian-games/turn-scores.jsonl
```

## 1. Game-kind and result stats

5,099 finished games. Kinds: `evaluation` (bot match / tournament) 4,548 (89.2%),
`human` (person vs bot) 483, `selfplay` 66, `versus` (person vs person) 2.

| slice | n | blue | red | draw | rated |
|---|---|---|---|---|---|
| all | 5,099 | 2,315 (45.4%) | 2,784 (54.6%) | **0** | 4,295 |
| evaluation | 4,548 | 2,083 (45.8%) | 2,465 (54.2%) | 0 | 4,295 |
| human | 483 | 211 (43.7%) | 272 (56.3%) | 0 | 0 |
| selfplay | 66 | 21 (31.8%) | 45 (68.2%) | 0 | 0 |
| tournament games | 766 | 350 (45.7%) | 416 (54.3%) | 0 | 725 |

Key facts:

* **Zero draws, zero voids in the entire export.** The scoring rule (running sum of
  area) makes ties practically impossible; close games (< 30-point margin) are 2.9%
  of decisive games. Treat every game as win/lose.
* Termination: 4,905 played to the end; **153 resignations — every single one by a
  human** (152 `human`, 1 `versus`); **41 forfeits — all in `evaluation` games**
  (see §7). No abandonment/adjudicated/void games exist in the current export.
* Rated games (4,295) are exactly the non-forfeit bot-vs-bot games; ratings never
  move on games vs people or self-play. Site totals: 81 ranked bots, 1,052 rated
  matches.
* Game length: median 120 moves (61 turns); only 194 games end early, all via a
  termination. Human games average 97.9 moves (resignations). Wall clock p50 12.6
  min, p90 55.5 min.
* In human games the human picks a color: humans won 38.3% as Blue but 49.5% as Red
  — humans also do better as Red (§2).
* Omni is **not yet on the site**: no `omni`-named bot in `/api/bots` or
  `/api/rankings`. Scout (our fork's parent) is on the rankings at Elo 1068.

## 2. Color balance: Red is the better color

Blue moves first but Blue's first turn is **one move** (rules, rule 2); everyone
else gets two. The data says that first move is a net handicap, not an advantage:

* Red wins **54.6%** of all games; rated-only 53.9%; tournament 54.3%.
* Self-play (same bot both colors — the cleanest color signal): Red 68.2% (n=66).
* Bot matches are color-paired by design: of 1,102 match pairs with both games
  present, 1,102 swap colors and 1,090 use the same opening — so per-bot
  color splits are measured fairly. Across independent bots the pattern is
  systematic: strong bots win *more as Red* (blueEdge = winrate-blue − winrate-red):
  YshanV1-Lite −9.6%, Riposte v8 −38.7%, AngelBot −37.8%, Great Barrier 2.0 −22.7%,
  iviazovetskyi_bot-V2-flash −25.5%. A handful of weaker bots show the opposite,
  but the top of the table is uniformly Red-leaning.
* Median score trajectory (decisive games): Red leads from turn 10 (40.5 vs 36.4)
  and the median lead widens monotonically to −229 by turn 61.

Consequences: nothing to exploit in matches/tournaments (colors are paired), but
(a) when a human picks a color against us, Red is the stronger side and more humans
win as Red; (b) eval and testing must not assume moving first is good; (c) our
tune A/B alternating colors is the right control.

## 3. Color-balanced bot leaderboard

Mined from the 4,548 bot-vs-bot games (no self-play), split by color, joined with
site Elo from `/api/rankings`. `balanced` = mean of winrates as Blue and as Red
(draws 0.5, but there are none). Minimum 20 mined games. Sorted by site Elo.

| bot | Elo | Blue W-D-L | Red W-D-L | wr Blue | wr Red | balanced | blueEdge |
|---|---|---|---|---|---|---|---|
| YshanV1 (hosted, GPU αZero-style) | 2196 | 79-0-2 | 77-0-2 | 97.5% | 97.5% | 97.5% | +0.1% |
| **YshanTest (wasm)** | 2060 | 57-0-2 | 59-0-0 | 96.6% | 100% | 98.3% | −3.4% |
| iviazovetskyi_bot-V2 (hosted) | 1934 | 42-0-4 | 42-0-4 | 91.3% | 91.3% | 91.3% | 0.0% |
| iviazovetskyi_bot-V2-ultra (hosted) | 1875 | 14-0-1 | 14-0-1 | 93.3% | 93.3% | 93.3% | 0.0% |
| FlyingTankOne (hosted) | 1760 | 14-0-6 | 18-0-2 | 70.0% | 90.0% | 80.0% | −20.0% |
| YshanV1-Lite (wasm) | 1727 | 118-0-95 | 139-0-75 | 55.4% | 65.0% | 60.2% | −9.6% |
| john.fun neural v3 (hosted) | 1727 | 15-0-12 | 17-0-8 | 55.6% | 68.0% | 61.8% | −12.4% |
| Hydra-1 (hosted) | 1658 | 8-0-5 | 9-0-4 | 61.5% | 69.2% | 65.4% | −7.7% |
| john.fun neural v4 (hosted) | 1642 | 22-0-12 | 21-0-13 | 64.7% | 61.8% | 63.2% | +2.9% |
| iviazovetskyi_bot-V2-flash (wasm) | 1635 | 17-0-20 | 25-0-10 | 45.9% | 71.4% | 58.7% | −25.5% |
| YshanV1-Lite-PUCT (wasm) | 1634 | 48-0-51 | 69-0-31 | 48.5% | 69.0% | 58.7% | −20.5% |
| Meteor-V3 (wasm) | 1615 | 14-0-10 | 16-0-8 | 58.3% | 66.7% | 62.5% | −8.3% |
| xmynetbot (wasm) | 1611 | 21-0-13 | 24-0-12 | 61.8% | 66.7% | 64.2% | −4.9% |
| AngelBot v2 WASM | 1607 | 32-0-23 | 32-0-23 | 58.2% | 58.2% | 58.2% | 0.0% |
| Great Barrier | 1598 | 244-0-94 | 258-0-81 | 72.2% | 76.1% | 74.1% | −3.9% |
| medium capybara net v5 (wasm) | 1576 | 173-0-134 | 195-0-111 | 56.4% | 63.7% | 60.0% | −7.4% |
| small baby search bot v5 | 1566 | 40-0-37 | 39-0-37 | 51.9% | 51.3% | 51.6% | +0.6% |
| … (34 more bots ≥ 20 games) | | | | | | | |
| **Scout (site bot, our fork's parent)** | **1068** | 38-0-138 | 54-0-121 | 21.6% | 30.9% | 26.2% | −9.3% |

Readings:

* The **strongest WebAssembly bot is YshanTest at Elo 2060 (98.3% balanced)** — a
  wasm NN bot within ~140 Elo of the best hosted bot. The wasm 5 s / 256 MiB limits
  are not a ceiling below ~2000 Elo.
* Every bot above ~1650 Elo is a neural-network bot. Pure search bots cluster at
  1450–1660 (Great Barrier 1598 is the best high-volume search bot, 74.1% balanced
  over 677 mined games).
* Scout is weak: 26.2% balanced. Omni (Scout fork + ID alpha-beta) starts from that
  baseline; the site-tested move-deadline work already shipped (see
  `OMNI_SEARCH_CHANGES.md`).
* `blueEdge` again skews negative down the whole table (§2).

## 4. Opening win rates

Bot-vs-bot games never start from the plain start: **the site chooses a 1-move
opening (Blue's first move) for every match and tournament pairing — 4,614 of 4,614
bot-vs-bot games, 69 distinct openings, always exactly one move.** So there is no
"choose a better first move as Blue" lever in rated play; the levers are Red's reply
(move 2, always ours in half our games) and Blue's second action (move 3).

Top openings by frequency (blue's winrate; pairings are color-swapped with the
same opening, so cross-opening differences are opening-driven, mixed over bot
pairs):

| opening (Blue's move 1) | n | blue win | red win |
|---|---|---|---|
| D10-E7 | 219 | 46.1% | 53.9% |
| D10-C8 | 140 | 44.3% | 55.7% |
| D10-D13 | 134 | 47.0% | 53.0% |
| A10-B7 | 117 | 47.0% | 53.0% |
| D10-B9 | 112 | **41.1%** | 58.9% |
| D10-D9 | 102 | 48.0% | 52.0% |
| A10-B9 | 101 | 49.5% | 50.5% |
| A10-C11 | 98 | **41.8%** | 58.2% |
| A10-D8 | 94 | **52.1%** | 47.9% |
| A10-C9 | 92 | 46.7% | 53.3% |
| D10-G11 | 88 | 45.5% | 54.5% |
| A10-D12 | 86 | 48.8% | 51.2% |

Spread across common openings is real but modest (≈ 11 points between A10-D8 52.1%
and D10-B9 41.1%); remember the ~54% Red baseline when reading these.

Human games (no chosen opening; the human picks Blue's move 1 — opponent mix is
uncontrolled, treat as directional): D10-A7 37.9% blue (n=116, bad), D10-G12 59.4%
(n=32, good), D10-F11 8.3% (n=12, terrible). D10-A7 and D10-F11 look like early
area-grabs that predictably lose (§5).

## 5. Score-dynamics findings

From the 5,099 full replays (per-turn cumulative scores; rule 8: each turn-end adds
both players' enclosed area to their score — score is a running sum of area):

* **Area compounds.** Median total per-turn score increment: ~9 points around
  turn 6 → 36 at turn 16 → 82 at turn 31 → ~100 at turn 46 → ~106 at turn 56.
  Final area is the "annuity" (winner p50 72.0 vs loser 29.5), and the score is its
  running sum: median final 1,909 (Blue) / 2,125 (Red), p90 ~4,600.
* **Early leads are anti-predictive.** The player ahead at turn 10 wins only
  **41.6%** of the time; turn 20: 48.2%; turn 30: 65.4%; turn 40: 86.8%; turn 50:
  95.2%; turn 60: 99.5%. Closing loops early (area now) is a trap; the game is
  decided in the mid-game, where room and cut tactics convert.
* **The last lead change is median turn 23 of 61** (p25 = 6, p75 = 35); 9.5% of
  games flip at turn ≥ 45 — never early-exit thinking, and never trust a "won"
  position at turn 20.
* **Margins blow out.** Decisive-game final margin: p10 114, p25 329, p50 867,
  p75 1,884, p90 2,772 points. 76.8% of games are ≥ 300-point blowouts; only
  2.9% finish within 30 points. Combined with 0 draws: there is no such thing as a
  safe small lead or a meaningful draw-avoidance strategy.
* Turn 1 is always 0.0–0.0 (Blue's single opening move never encloses area);
  Blue's one-move first turn is the tempo handicap behind §2.

Eval consequences for Omni: under-weight current area early, keep the room term
strong through turn ~25, and treat cut/defense as the mid-game currency. The
tuner's position sampling should bias toward turns 20–45, where games are actually
decided.

## 6. Top-5 ranked improvement ideas (effort vs Elo)

Ranked by expected Elo per unit effort, grounded in §1–§5 and the current repo
state (Omni = Scout fork + ID alpha-beta + deadline guard; `bots/meridian/nn` =
trained supervised value/policy net, wasm-ready, with `nnbench`).

| # | idea | effort | expected Elo | why / evidence |
|---|---|---|---|---|
| 1 | **Hybrid: NN eval + policy priors inside the deployed ID search.** Wire `nn::eval` (value tanh(margin/1500), ~400 KB weights vs 8 MB limit) as the leaf eval, and the 17,328-way policy head for move ordering / candidate pruning. | M | **+200–400** (the one big lever) | Every bot above ~1650 Elo is neural. The strongest *wasm* bot, YshanTest (2060), is an NN in our exact deployment regime. Policy-first ordering also fixes the known deep-search weakness (capped, length-ordered replies miss cut refutations — Setup A in OMNI_SEARCH_CHANGES.md). The deadline guard already makes any added leaf cost safe. |
| 2 | **Stay forfeit-proof (reliability).** No panics/allocations outside caps on the reply path; TT stays capped (2^19); stderr < 16 KB/request; keep the 4300 ms deadline target (measured worst wasm request 4305 ms, 195 ms under the failure line). If ever hosted: one polling loop, heartbeat, reply inside lease, sane `maxConcurrentGames`. | S | **protects ~5–10% of tournament points** | 41 of 766 tournament games (5.3%) were forfeits; ClarkeBot forfeited 12; the 10/01 champion was *withdrawn on two strikes* (rounds missed / games forfeited). Zero forfeits is free rating. |
| 3 | **Opening book for Red's reply (move 2) and Blue's move 3.** The site always picks move 1 (§4), so book *our* answers to the 69 openings, biased away from the anti-predictive early-area replies; saves clock for turns 20–45. | S | **+10–30** | §4 spread (A10-D8 52.1% blue vs D10-B9 41.1% blue) and §5 (early leads lose). Cheap, purely additive to the shipped bot. |
| 4 | **Compounding-aware eval re-tune.** Re-run `tune`/A/B with position sampling biased to turns 20–45, re-check `room_weight` (0.4) and cut-extension depth under the NN eval, and prefer lines that keep room late over lines that bank area early. | S–M | **+20–60** | §5: leader at turn 10 wins 41.6%; last lead change median turn 23; margins blow out late. Current uniform-ply tuning under-weights the decisive window. |
| 5 | **Ship only SPRT winners through the in-repo arena.** Use the serve-side checkpoint ensemble + recency replay + SPRT champion gate (commit 7f4e57a) and timed A/B at the deployed 4300 ms/move (Setup B/C in OMNI_SEARCH_CHANGES.md) for every candidate above; upload only measured winners, one version at a time (3 versions per bot). | S (process) | **compounding** | The 1h-tune history shows noise-level candidates are common (width12 +6 over 120 games ≈ 1σ). This converts ideas 1/3/4 from guesses into Elo deltas, and version-rollback protects the live rating. |

Ceiling path (not in the top 5, hardware-gated): a hosted GPU bot à la YshanV1
(2196) — hosted leases allow ~25 s/move vs wasm's 5 s — but wasm NN bots already
reach 2060 and have won tournaments, so this is optional.

## 7. Jam and tournament facts

**Is there a jam / announced event? No.** As of 2026-10-20: `/api/tournaments`
lists only 5 completed tournaments (last finished 2026-10-02); there is **no
`draft`/upcoming entry**, `/api/jam` returns 404, `/jam` is just the SPA shell,
and the site's JS bundle contains no "jam", "deadline" or announcement strings.
Tournaments are created ad hoc by moderators ("New tournament" is a
moderator-only action; a `draft` status shows as "Upcoming" on the site). So:
**no published format deadline exists** — the readiness posture is "be uploaded
and reliable at any time", and watch `/api/tournaments` for a `draft` entry.

The five tournaments so far (all complete):

| name | created | entrants | format | budget | champion |
|---|---|---|---|---|---|
| Initial | 09-26 | 14 | 4 Swiss + top 4 | 72 games | VladNet |
| 9/27 | 09-27 | 25 | 6 Swiss + top 8 | 176 | Great Barrier |
| 9/28 | 09-29 | 35 | 4 Swiss + top 8 | 168 | medium capybara net v5 |
| 9/29 | 09-30 | 40 | 4 Swiss + top 8 | 192 | medium capybara net v5 |
| 10/01 | 10-02 | 40 | 4 Swiss + top 8 | 192 | iviazovetskyi_bot-V2-ultra (later withdrawn, two strikes) |

Site tournament rules (mined from the site's own text, `/tournament/:id` page):
each Swiss pairing plays **both colors with the same opening**; ties break by
Buchholz → head-to-head → seed; a playoff drawn match goes to the higher seed;
**tournament games count for Elo, except games lost by forfeit**. Format caps from
the UI: at most 8 Swiss rounds, playoff of 0/4/8, at most **200 games per
tournament** (a 40-bot, 4+8 event is 192 games). Withdrawal: "A bot that crashes,
plays an illegal move or goes offline loses that game. Two strikes (a round
missed, or one with a forfeit) withdraw a bot."

**Timeout-forfeit rules (bot-api.md, the exact clauses):**

* Since **2026-10-04**: "a move that runs past the limit [`limits.moveTimeMs`,
  5,000 ms in games] is stopped and counts as the bot's failure, in a tournament a
  forfeit" (before that date the stop was ~23 s browser / 25 s server).
* Tournament forfeit triggers: reporting an error, **letting a lease lapse
  twice**, playing an illegal move, or going offline mid-game; a bot offline when a
  round is paired sits that round out (and two strikes withdraw it).
* Analysis requests keep their `visits` budget and are stopped after 30 s
  (browser) / 25 s (server). Hosted job leases: reply before `leasedUntil`, at
  most 25 s after taking the job. A bot is online while it polls, offline 90 s
  after its last request.
* Other limits: wasm file 8 MB, memory 256 MiB, reply 1 MB, 3 versions per bot /
  32 MB per account, 10 bots per account; the bot's page shows median/slowest timed
  move and how many of its last 1,000 ran over the limit.
* Omni compliance: we aim for `min(86% × moveTimeMs, 4300 ms)` with clock checks
  every 32 nodes; measured worst shipped-wasm request 4305 ms, 0 illegal replies
  (every candidate comes from `Position::legal_moves()`).

**Can a wasm bot resign or surrender? No.** Three independent confirmations:

1. The reply schema in bot-api.md offers only `{ "move": id }`, `{ "analysis": … }`
   or `{ "error": "…" }` — for hosted and WebAssembly bots alike. There is no
   resign/surrender reply; a move must come from `state.legalMoves`.
2. The data: all **153 resignations in 5,099 games were resigners of type `human`**
   (152 person-vs-bot, 1 person-vs-person). Zero resignations in any
   `evaluation`/`selfplay` game. The site's Resign button exists only in the
   human-game UI.
3. The only bot "early exit" is an error/timeout → `termination: "forfeit"` (41
   games, all bot games): a certain **loss**, a tournament **strike**, and forfeit
   losses **do not count for Elo**.

**Verdict: a wasm bot cannot resign, and it should never fake one.** Replying with
an error to escape a lost position converts at best a loss into exactly the same
loss plus a strike (two strikes = withdrawn from the tournament). With zero draws
and 2.9% close finishes, playing every game out is both mandatory and free. The
correct "resign logic" is the one already shipped: never time out, never error,
always answer a legal move.

## 8. Fetch log and reproduction

All site fetches succeeded this pass: `meridian-games.zip` (via
`/downloads/meridian-games.zip`, the link in bot-api.md; the bare `/meridian-games.zip`
404s to the SPA shell), `/api/games/export` (describes the same file), `/bot-api.md`,
`/api/bots`, `/api/rankings`, `/api/tournaments`, and the SPA bundle
`/assets/index-CjbM1yye.js` for rules text. Failures: `/api/jam` 404 (no such
endpoint), `/jam` and `/tournaments` return the SPA shell (client-side routes) —
noted, not blocking. The one real incident: the prior scratch was lost (empty
`/tmp/opencode`), so everything above was re-fetched and re-computed from zero in
this pass.

Reproduce the mining: run `scores` (above), then
`python3 /tmp/opencode/meridian-games/analyze.py` → regenerates `analysis.txt`.
The zip refreshes every 6 h; `/api/games/export` reports the current one.
