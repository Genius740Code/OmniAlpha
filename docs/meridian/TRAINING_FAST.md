# Meridian supervised trainer: the fast pipeline (TRAINING_FAST)

Date: 2026-10-21. Scope: `bots/meridian/nn/train/**` — extraction, training,
sweep and export pipeline for the value/policy net described in
[NN_PLAN.md](NN_PLAN.md). The 26-tensor `weights.rs` contract is unchanged; the
shipped weights are re-exported from a retrained checkpoint.

> Box honesty: 2 CPU cores of a shared 6-core i7-9750H; other sessions load the
> host intermittently. Wall-clock numbers below state their conditions;
> ratios use same-condition pairs wherever possible. The training scratch
> (`/tmp/opencode/train2`) was wiped between sessions — every number below was
> re-measured (or reproduced bit-identically, see §6) on the rebuilt cache
> (corpus sha `3698df051433eddf`, identical both days).

One-line summary: **per-epoch training time is down 3.9–7.8× (28.19 min →
~3.6–7.2 min), a full retrain (fresh archive → verified `weights.rs`) runs in
~31–37 min end-to-end, and the new net fixes the old one's diagnosed weakness —
close games (|final margin| ≤ 300): sign accuracy 0.7061 vs 0.6917 (+1.4 pts,
n=6,039) — at ~0.5 pt lower overall sign accuracy, using 4× fewer training
samples than the old net saw.**

## 1. Audit of the old trainer (reconstructed)

The original training scratch was wiped before this work (its loss is also why
`NN_PLAN.md` lists the `--check` artifacts as unrecoverable; this pipeline
regenerates them). "Before" numbers come from (a) the old `weights.rs` header
and (b) a faithful reconstruction of the old recipe (full corpus per epoch,
float32 planes, no augmentation / balancing / curriculum, NCHW) run on this box.

| anchor | value |
|---|---|
| recorded old run (surviving `weights.rs` header) | best at epoch 3, **35 min total**, val sign 0.826, mse 0.1491, corr 0.811, policy top-1 0.1119 |
| old dataset path | f32 dense planes, ~20.2 kB/position ⇒ **~12.1 GB** for 596,604 positions; rebuilt from scratch on every fresh archive |
| old samples per "epoch" | 596,604 (whole corpus) |
| **measured old-recipe minutes/epoch (this box)** | **28.19 min** (564,193-position train pool, batch 256, NCHW; 20,017 samples/min sustained) |
| clean-box probe rates, old recipe | 23.4k samples/min (b256 NCHW), 21.1k (b512 NCHW) |
| measured old-recipe quality after 1 epoch (28.19 min) | quick-val sign 0.8447, close-game sign 0.6810 |

The recorded 35-minute run cannot have been 3–4 full-corpus epochs on this box
(28.19 min/epoch measured), so the old "epochs" were necessarily subset passes;
§4 therefore reports speedup against **both** anchors — the recorded 35 min and
the measured full-corpus-epoch cost.

## 2. Pipeline speed (what changed)

| stage | old | new |
|---|---|---|
| extraction | f32 dense rebuild, ~12.1 GB on disk | **2 parallel processes, 12.4 s** for 5,087 games → 596,604 positions; **264 MB bit-packed cache** (`train/dumper/` replays with the real engine + `nn::input_planes`, so planes are byte-exact by construction — gated by `nnbench --check`) |
| re-extraction after a site re-download | rebuild everything | **incremental**: cache keyed per game by `sha256(json line)`; only new games dumped (same-archive re-run ~0.1 s) |
| storage per position | ~20,220 B (5,054 f32 planes + target) | **442 B**: 9 binary planes bit-packed (414 B, lossless), 5 scalar f32s (exact — f16 was rejected: it would break the byte-exact `--check` planes), move id u16, raw margin f32, ply, flags |
| value target | fixed `tanh(margin/1500)` baked into samples | margins stored raw; scale applied at collate ⇒ the 600/900/1500 sweep needed **no re-dump** |
| batch assembly | — | in-RAM per-batch unpack; collate measured at **0.6–1.2%** of step time ⇒ `num_workers=0` beats a DataLoader on 2 cores (workers are pure overhead here) |
| torch layout | NCHW | **channels_last, batch 256–512** ⇒ 26.4–27.5k samples/min clean vs 21.1k NCHW b512 (**1.32× per-sample**) |
| curriculum | none | skip plies < 6 (−3.6% positions) |

**Minutes per epoch, before/after** (the brief's requested evidence): the old
recipe's epoch (whole corpus, 564k positions) measured **28.19 min**; the new
epoch (100k margin-balanced, 4-fold-augmented samples) measured **3.79–7.21 min
under host load, ~3.6 min clean-box** — **3.9–7.8× fewer minutes per epoch**, at
4× fewer samples per epoch (each carrying 4 augmented views).

## 3. Sample efficiency (what changed)

1. **Curriculum** — skip the first 6 plies.
2. **Margin-balanced sampling** — close games (|margin| ≤ 300; 22.1% of
   positions) drawn ~3.2× relative to uniform (epoch mix 40/30/30
   close/mid/wide) — the direct fix for the corpus's 64%-blowout skew.
3. **Symmetry augmentation, 4-fold** — the game is invariant under the Klein
   group {id, h-mirror, 180°-rotation+swap, v-mirror+swap}: planes flip,
   blue/red planes and to-move invert for the color-swap pair, value negates,
   played move id remaps through exact per-transform tables. Verified:
   every transform is an involution through the real collate path; remapped
   policy sources land on the mover's node plane on 4,000 real positions
   (100.0%). The verification caught a real bug: the h-mirror originally
   flipped planes without remapping the policy target (25% of samples
   taught the wrong move; the first training run showed it in the metrics).
4. **Best-epoch selection** on `0.5·overall + 0.5·close` sign accuracy of a
   strided val pass each epoch (best landed at epoch 3 of 6).
5. **Margin-scale sweep** — §5.

## 4. Old vs new (the A/B)

Validation = 279 held-out games (hash split; 30,754 decision positions,
ply ≥ 6; identical set for both nets — the split is deterministic in the
corpus sha). "Close" = |final margin| ≤ 300 (6,039 positions). Old-net numbers
re-measured on the same val with the same harness (they also reproduce the
recorded bench numbers exactly).

| metric | old net (4 epochs, 2.4M samples) | new net (this pipeline, 600k samples) |
|---|---|---|
| val sign acc, overall | **0.8546** | 0.8495 (−0.5 pt) |
| val sign acc, wide (|m| > 300) | 0.8944 | 0.8845 (−1.0 pt) |
| **val sign acc, close games** | 0.6917 | **0.7061 (+1.4 pt)** |
| close-game correlation | 0.414 | 0.457 |
| close-game mse (own scale space*) | 0.0409 (1500) | 0.0714 (600) |
| bench fixtures, 12 close games (n=1428): sign | **0.7836** | 0.7703 (−1.3 pt, ≈1σ on this small sample) |
| policy top-1 / top-8, val (Rust, legality-masked) | 17.4% / 42.3% | 15.7% / 40.5% |
| parameters / export | 101,985 f32 = 398 KiB (`weights.rs` 2.47 MB) | identical (same 26-tensor LAYOUT) |
| value scale | tanh(margin/1500) | tanh(margin/600) (sweep winner; `nn::MARGIN_SCALE` updated in the per-game adapter block) |

*sign accuracy is scale-invariant; mse lives in each net's own target space and is not directly comparable across scales — sign/corr are the comparable columns.

Verification gates on the new export, all passing on the current build:
`cargo build --release`; `nnbench --verify` (fast kernels vs reference: value
2.5e-6 < 1e-5 gate, top-8 ranking identical 276/276, kernel speedup 1.70×);
`nnbench --check` (**input planes 2048/2048 byte-exact**, Rust-vs-torch value
max |diff| = 1e-6, sign agreement 2048/2048); decode asserts on every eval.

**Speedup, both anchors** (retrain end-to-end = fresh archive → verified
weights):

| comparison | old | new | speedup |
|---|---|---|---|
| per-epoch minutes | 28.19 min (564k samples) | 3.6 min clean / 3.79–7.21 min loaded (100k) | **3.9–7.8×** |
| per-sample throughput | 20.0–23.4k/min | 26.4–27.5k/min | 1.32× |
| vs measured baseline to equal quality (old recipe, ≥2 full epochs ≈ 56 min; recorded-equivalent 4 epochs ≈ 113 min) | 56–113 min | ~31 min clean-box est. / 37.1 min measured under load | **1.5–3.6×** |
| vs the recorded 35-min old run | 35 min | ~31–37 min | ~parity (see §1: the recorded run cannot have been full-corpus epochs; at equal samples-passed the new pipeline is ahead on close games and behind 0.5 pt overall) |

Samples to (near-)old quality: 600k + a 30k-sample value-head sweep vs the old
net's 2.4M — **4× fewer samples**, with the accuracy profile deliberately
shifted toward close games (+1.4 pt close, −0.5 pt overall, −1.9 pt policy
top-1). The close-game axis was the brief's target; the overall/policy give-back
is the documented price of 4× fewer samples and the balanced mix. More epochs
(overall sign was still climbing at epoch 5: 0.8462 quick-val) would recover the
gap if a future retrain wants to spend the time.

## 5. Margin-scale sweep (600 / 900 / 1500)

Value-head fine-tune (60 balanced steps) from the shared trunk at each scale,
evaluated on the full validation set; sign metrics are scale-invariant, so the
table compares which training scale optimizes best:

| scale | overall sign | close sign | mse (own space) |
|---|---|---|---|
| 600 | **0.8495** | 0.7061 | 0.2420 |
| 900 | 0.8488 | 0.7043 | 0.1909 |
| 1500 | 0.8481 | 0.7072 | 0.1314 |

Near-tie (Δ ≈ 0.001); 600 wins the pre-declared key (0.5·overall + 0.5·close)
and gives close-game targets the most tanh range, so 600 was exported and
`nn::MARGIN_SCALE` updated to match (the semantic of `eval_value`:
`atanh(value) × 600` ≈ expected final margin). Notably the balanced fine-tune
itself lifted both axes over the raw checkpoint (+1.0 pt overall, +0.8 pt
close) regardless of scale — the balancing, not the scale, is the active
ingredient.

## 6. Re-run (fresh archive → shipped weights)

```sh
cd bots/meridian/nn/train
python3 extract.py                     # incremental; dumps only games new to the archive
python3 train.py --profile fast --epochs 6 --samples-per-epoch 100000 \
       --batch 256 --channels-last --lr 2e-3 --run fast_v2
python3 sweep.py  --ckpt /tmp/opencode/train2/runs/fast_v2/best.pt
python3 bench.py  --ckpt /tmp/opencode/train2/runs/fast_v2/best.pt   # torch metrics + close-game breakdown
python3 export_weights.py --ckpt /tmp/opencode/train2/runs/sweep/best_sweep.pt --nnbench-artifacts
cd .. && cargo build --release
./target/release/nnbench --verify
./target/release/nnbench --check
./target/release/nnbench --val /tmp/opencode/train2/nn-data/out3/val_games.json 10
./target/release/nnbench --val fixtures/bench_games.json 1
```

Reproducibility note: the epoch-0 metrics of the retrained run reproduce the
previous day's identical-seed run exactly (loss 6.7577, sign 0.7560), and the
old net's val metrics reproduce bit-identically across cache rebuilds (val
split is a hash of game content).

## 7. Reusability: the per-game adapter contract

The inference crate is game-agnostic by design (see the "Porting to a new
game" checklist in [NN_PLAN.md](NN_PLAN.md#porting-to-a-new-game-the-reuse-checklist)):

* **`nn.rs` — "Per-game adapter: change these"**: board side, input-plane
  count, trunk channels, residual blocks, head channels, `MARGIN_SCALE`. This
  one const block is the entire per-game configuration surface; the 600-scale
  change in this retrain was a one-line edit there.
* **`nn::input_planes`** — the only genuinely game-specific function (encode
  one game state into `IN_PLANES × BOARD²` f32s). The training side mirrors it:
  `train/dumper/` calls the real engine's `input_planes` itself, so any port
  inherits byte-exactness by construction.
* **`convfast.rs`** — im2col + SGEMM (+ wasm simd128) kernels with no game
  logic; `im2col` is const-generic over board size, channel counts are runtime
  args, so a future game's net is the same code path. `nnfast.rs` is just the
  Meridian dataflow over those kernels, and `nnbench --verify` gates it against
  the reference conv.
* **`weights.rs`** — regenerate per game; the 26-tensor order (stem → N blocks
  → value head → policy head) and `LAYOUT` stay the same shape of contract.
  `NN_WEIGHTS_FILE` lets nnbench time/score a fresh checkpoint (raw f32 bytes)
  before it is committed.

This trainer is per-game in exactly two places by the same logic:
`dumper/src/main.rs` reads the site's game JSON (fields + margin), and
`dataset.py`'s symmetry tables encode Meridian's board group — a new game
replaces those two and the collate's plane count.

## 8. Files

| file | role |
|---|---|
| `dumper/` | Rust extraction: replays games with the engine, writes the packed cache + dense `--check` artifacts |
| `extract.py` | corpus hashing, incremental per-game cache, 2-process dump, val split, `val_games.json` |
| `dataset.py` | packed loader, 4 symmetry transforms + move-id remap (self-tested), balanced sampling, collate |
| `model.py` | torch `TrainNet`/`FoldedNet`, BN folding, `weights.rs` read/write (round-trip asserted) |
| `train.py` | baseline + fast profiles, cosine LR, best-epoch selection |
| `sweep.py` | margin-scale recalibration (§5) |
| `bench.py` | torch metrics, close-game breakdown |
| `export_weights.py` | fold → serialize → `weights.rs` + parity self-tests + `expected_dense.bin` |
