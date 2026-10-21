# HANDOFF — Meridian jam work, what remains (Oct 21 2026)

Read `OMNI_SEARCH_CHANGES.md`, `NN_PLAN.md`, `TRAINING_FAST.md`, `JAM_INTEL.md` first. This file = current state + the ONE unfinished task.

## Goal

Strongest bot for the Meridian strategy jam (constellation.blueshrimp.eu — .uk, bot API in `/home/genius74o/omnizero/bots/meridian`). Artifact = `bots/meridian/omni.wasm`. Jam readiness = uploaded + zero forfeits; no jam currently scheduled (check `/api/tournaments`).

## Shipped state (all committed, pushed; master @ 9cfadc4)

| Piece | State |
|---|---|
| omni.wasm (270,773 B, sha `8ec84dc8…`) | Scout-exact 2-ply + moveTimeMs awareness + regression fix. Worst wasm move 50-125 ms. **This is what the site should run.** |
| Deadline guard | In search.rs, verified: worst overshoot 0.55 ms, whatever you do later must stay under 4,300 ms (5,000 limit, safety margin). |
| Deep/ID search | Library code in search.rs. **LOST A/B (10-32-0). Do not ship, do not revive.** Reason: capped reply sets miss quiet refutations. |
| NN crate `bots/meridian/nn` | Trained on 5,087 games. Value 78.4% sign acc (vs 56% heuristic), close games 70.6%. Value eval 4.96 ms wasm / 6.0 ms native (im2col→SGEMM, wasm SIMD). All gates pass (`nnbench --verify`, `--check`). MARGIN_SCALE=600 in nn.rs adapter block. |
| Fast training | `nn/train/`: 31 min end-to-end retrain, sha-keyed cache, margin-balanced sampling. Docs in TRAINING_FAST.md. |
| Framework extras | `tools/solve_oracle` exact solver, Bradley-Terry Elo in `python/gai/sprt.py`, 46/46 pytest (run `PYTHONPATH=python pytest python/tests -q`). |
| A/B harness | `omni/src/bin/tune` (+ `bench`, `parity`, `scores`). Tuning history + A/B table in OMNI_SEARCH_CHANGES.md. |

## THE REMAINING TASK — NN value re-ranker (designed, never wired)

Integration agent was stopped by owner. The plan it was about to execute:

1. **Design**: after `analyze_2ply` returns candidates, re-rank ONLY the top band (8-16 candidates within K points of best — pick K by looking at eval spread over real positions in `omni/target/games.jsonl`, exists from earlier work). Evaluate NN value head on position AFTER candidate + best reply (the same depth Scout's eval saw). Select best by NN. **Never introduce new moves** (that's the capped-reply failure mode).
2. **Calibration**: NN unit = expected margin / 600. So NN×600 ≈ points; validate on ~100 real positions vs 2-ply eval before trusting the ordering conversion.
3. **Wiring**: path dep `meridian-nn = { path = "../nn" }` in omni Cargo.toml (nn crate already lib + rlib). Hard budget: max BAND NN evals (~80 ms wasm). Analysis requests (visits < ~64): leave Scout-exact. Thread the existing `Limits.deadline` through the NN path.
4. **GATE (mandatory)**: extended tune A/B, re-ranker ON vs OFF, same opening rotations, ≥24 games. Ship only if ON clearly wins (~≥60% of decided games). If it loses → revert, document, stop. This rule is non-negotiable: three bigger-search attempts lost their A/B already.
5. **If accepted**: `cd omni && cargo build --release --target wasm32-wasip1 && cp target/wasm32-wasip1/release/omni.wasm ../omni.wasm`, re-run `./target/release/smoke` + wasm harness (see OMNI_SEARCH_CHANGES.md reproduce block; a fixed Node harness is at /tmp/opencode/nn-wasirun.js, may need rebuilding into /tmp or into the repo), worst timed move must be < 4.6 s.
6. **Docs**: update OMNI_SEARCH_CHANGES.md + bots/meridian/README.md either way (ACCEPT or REJECT with the A/B numbers).

## Publishing (manual, needs owner's session — routes take browser session, not bot token)

1. Copy new omni.wasm to Windows: `cp bots/meridian/omni.wasm /mnt/c/Users/fese/Downloads/omni.wasm` (WSL mounted).
2. My bots → WebAssembly → upload; verify SHA-256 matches `sha256sum bots/meridian/omni.wasm`.
3. Tick "This bot can analyze positions"; set maxConcurrentGames 2-3.
4. Check bot page: median ~1.5-2 s, overridden count 0. Sanity match vs Scout first.

## Next-round ideas (after re-ranker ships, ranked from JAM_INTEL.md)

1. Re-train when archive grows (refresh every 6 h; human games improve close-game calibration — the 89% bot-vs-bot corpus is the main cap).
2. Value head in search is still infeasible (needs all 4096 leaves; ~5 ms × 4096 = 20 s). Only if a ≥10× further wasm win lands (AVX-512-native-style work or pruned value-only net).
3. Opening book for Red reply + Blue move 3 (site always picks move 1; only our replies need coverage). `solve_oracle` can label them exactly.
4. Compounding eval re-tune on decisive turns 20-45 sampled positions (early leads are anti-predictive: leader at turn 10 wins 41.6% only).

## Gotchas (learned the hard way)

- Wasm SIMD needs `RUSTFLAGS="-C target-feature=+simd128"`; plain build must stay working (site's Node/V8 + wasmtime both fine either way; plain-wasm is 1.9× not 5.7×).
- Do NOT delete or reshape the 26-tensor LAYOUT contract in weights.rs; nnbench asserts it.
- 2-core box: any A/B must finish in ≤ ~20 min wall time or it will be abandoned by timeouts.
- Sessions occasionally die on API timeouts — relaunch fresh, state is on disk, brief said agent must read files first.
- /tmp/opencode scratch is wiped on reboot; anything you need long-term must live in the repo.
