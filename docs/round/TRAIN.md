# Round C — Train-side (python/gai) report

Agent C of 3 (GLM-5.3-flash), 2026-10-21. Scope: **Elo per wall-clock minute in the
training loop**. Files touched: `python/gai/{train,orchestrator,data}.py`,
`python/tests/{conftest.py,test_train_ema.py,test_replay_dup.py}`,
`results/roundC/**`. Nothing under engine/**, tools/**, configs/** was modified.
No git commits.

**Test gate: `python3 -m pytest python/tests -q` → 61 passed** (46 pre-existing + 15
new; the instructed repo-root invocation now works via the added
`python/tests/conftest.py` that puts `python/` on sys.path — previously it only
collected when run from `python/`).

## Experiments (one knob at a time, measured)

### 1. EMA/SWA export — SHIPPED (flag default OFF), measured NEUTRAL at 3-min scale

`training.ema_decay` (e.g. 0.99): `make_trainer` maintains an exponential moving
average of parameters after every optimizer step (BatchNorm buffers copied, not
averaged). EMA weights are exported as `latest_ema.pt` / `latest_ema.ts` at
checkpoint cadence + once at run end (per-iter export measured 32 ms MLP /
186 ms ResNet — too expensive every iter; staleness ≤ checkpoint interval is fine
because EMA moves slowly). Default `ema_decay: 0` → exactly the old code path.

Pilot A (3 min, champion vw2 regime: mlp128x2, vw=2.0, lr 1e-3, batch 256,
threads 2 selfplay + `num_threads: 1` train, ema_decay 0.99, 291 iters):

- Arena regular-vs-EMA, 200 games @50 sims: **84-83-33, +1.7 ±44** → even.
- (40-game warmup: 18-13-9, +43.7 ±98 — same conclusion.)
- Verdict: EMA = same strength at 600 training steps. At 60-min scale (~6k steps)
  EMA may matter more; needs a 60-min A/B before enabling in competition configs.

### 2. LR schedule — verified existing, small API improvement

Warmup+cosine (`lr_schedule: cosine`, `warmup_frac`) and the final low-LR anneal
(`anneal_frac`/`anneal_factor`) already existed. Extracted the inline cosine math
into `gai.train.cosine_lr(base, frac, warmup_frac, min_lr_frac)` — a pure,
unit-tested function; orchestrator now calls it. New optional knob
`training.min_lr_frac` (default 0.02 = the previously hardcoded floor; behavior
identical unless set). Values verified: frac=0→0, warmup midpoint→base/2,
frac=1→base*min_lr_frac, monotone decay after warmup.

### 3. Steps/iter balance — CONFIRMED HEALTHY, train is NOT the bottleneck

Both 3-min pilots: **effective_steps mean 10.6–10.9, mode 11** (range 8–13; 8-9
only in the first iters while the buffer fills). The v9 starvation (sustained
8/iter) is gone; the v10-restored regime holds at 150 sims / 24 games / reuse 3.0.

Time split per iter (pilot A, 2-core box): selfplay 0.32 s (86%), train step
0.023 s (6%), remainder = holdout eval + exports. New log fields make this visible
every iter: `train_s`, `selfplay_s`, `samples_per_s`, `torch_threads`,
`dup_weight`, `effective_steps`. b4-style "smaller batch + more steps" is
unjustified while train is ~6% of wall-clock — the lever is selfplay (not mine).

### 4. Duplicate downweighting (TODO #4 remainder) — SHIPPED (flag default OFF), POSITIVE TREND

`training.dup_weight` (0..1): ReplayBuffer now tracks a content hash per slot
(vectorised FNV over (x,pi,z) bytes) and the number of active slots per content.
`sample(dup_weight=w)` draws with probability ∝ count^(−w) via rejection sampling
(exact target distribution; acceptance vector cached across the 12 steps and
invalidated on add). `dup_weight=1` gives each distinct position equal total mass
regardless of how many copies exist (transpositions, repeated openings);
`dup_weight=0` (default) is the untouched uniform path. Construction takes
`track_dups=` — with the flag off, add()/sample() are bit-identical to before
(covered by tests).

Cost (measured, 1200 fresh rows/iter into 200k cap):

- add(): 0.43 → 1.86 ms/iter when tracking (hashing); flag off → 0.43 (unchanged).
- sample: uniform 0.09 ms vs dup-weighted 0.12–0.15 ms per step (cached); per iter
  +3.6 ms train time measured in the clean pilot (+15% train, ~+1% total iter).

Pilot C (3 min, identical to pilot A except `ema_decay: 0`, `dup_weight: 1.0`;
280 iters vs 291 — within noise):

- Arena pilotC-vs-pilotA, 200 games @50 sims: **109-64-27, +79.5 ±46**
- Arena pilotC-vs-pilotA, 200 games @100 sims: **101-73-26, +49.0 ±46**

Both positive; +79.5 clears its CI at 50 sims, +49 is borderline at 100. Single
run pair — historical 3-min/10-min run-to-run spread (−35..+107) means this is a
**promising signal, not an established gain**. Recommendation: next 60-min run
sets `training.dup_weight: 1.0` in an A/B (it stacks naturally with vw2; no other
code changes needed). Held-out last-30: policy 1.036 vs 1.097 (better), value
0.568 vs 0.480 (worse) — mixed, within noise.

### 5. PyTorch CPU perf — `training.num_threads` SHIPPED; 1 thread wins for MLP

Micro-bench (`results/roundC/bench_threads.py`, batch 256, 40 steps):

| arch | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| mlp 128x2 (champion vw2) | **121.7k samples/s** | 76.5k | 67.5k |
| resnet 32c x3 (v9) | 3.7k | **5.4k** | 3.0k |

**MLP trains 1.59x faster single-threaded** (thread sync swamps small matmuls on
2 cores); ResNet prefers 2. `training.num_threads` sets intra-op threads (+
interop=1 + flush-denormal); `0`/absent = untouched defaults. Both pilots ran
`num_threads: 1` and logged samples/s ≈ 117–122k, matching the micro-bench. Suggest
`num_threads: 1` for MLP configs, 2 for ResNet configs (next agent's choice — I do
not own configs/**).

## Other notes

- **Pilot B validity caveat:** the first dup_weight pilot (results/roundC/pilotB)
  coincided with external box load (selfplay +40% slower); pilot C is the clean
  rerun and the one cited above. Kept for the record.
- **Arena tooling note:** `./build-lt/evaluate nn:<ts>` loads fine with no
  LD_LIBRARY_PATH (torch RPATH resolves); 200 games @50 sims MLP-vs-MLP ≈ 2 s.
  Cheap arenas make 200+-game gates standard for train-side A/Bs.
- `docs/HANDOFF.md` and `tools/arena_g.cpp` show as modified/untracked in git —
  NOT from this agent (parallel agents A/B).
- Configs I used live in `results/roundC/pilot{A,B}.yaml` (I don't own configs/**).

## Suggested config deltas for the next 60-min run (for whoever owns configs)

```yaml
training:
  num_threads: 1        # MLP: 1.59x train throughput
  dup_weight: 1.0       # promising (+79.5/+49 over 200 games, 3-min pilots); A/B it
  # ema_decay: 0.99     # neutral at 3-min; only with a 60-min A/B
```

## New flags summary (all default-OFF → zero behavior change)

| flag | where | effect |
|---|---|---|
| `training.ema_decay` | train.py/orchestrator.py | EMA weights; exports latest_ema.pt/.ts at checkpoint cadence + run end |
| `training.num_threads` | train.py/orchestrator.py | torch intra-op thread cap (+interop 1, flush denormal) |
| `training.dup_weight` | data.py/orchestrator.py | duplicate downweighted sampling (needs track_dups, auto-enabled) |
| `training.min_lr_frac` | train.py/orchestrator.py | cosine floor (default 0.02 = old hardcoded value) |

New log.jsonl fields per iter: `train_s`, `selfplay_s`, `samples_per_s`,
`torch_threads`, `dup_weight` (+ existing `effective_steps`, `recency_weight`).

## Tests added (15)

- `test_train_ema.py` (7): EMA disabled→None, one-step blend math, decay<raw
  movement, export roundtrip, ResNet buffers copied not averaged, cosine_lr values
  + monotonicity + floor, setup_torch_threads noop/set.
- `test_replay_dup.py` (7): same-content counts, distinct rows, ring-overwrite
  retirement, wraparound invariants (per-slot hash == content hash), weighted
  distribution (statistical, seeded), flag-off bit-identical paths, hash≠0.
- `conftest.py`: enables `python3 -m pytest python/tests -q` from repo root.
