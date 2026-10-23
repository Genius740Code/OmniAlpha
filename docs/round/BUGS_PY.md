# Round bug-hunt — Python scope (agent A, narrowed)

Scope after the update: `python/gai/**`, `python/tests/**` only. C++ engine findings
from the first half of this session were handed to agent D (advance_root proven-sign
flip, shortcut subtree reuse, resign-probe scoring, batch dedupe) and are NOT repeated
here except where a Python-side test pins the behavior.

All evidence below is live-verified on this box (build/selfplay present, CPU torch
2.13.0+cpu). Suite: `python3 -m pytest python/tests -q` → **83 passed** (was 61
before this round; +22 tests, 0 pre-existing tests broken).

---

## 1. Value-target sign audit — NO LIVE BUG (verified bit-exact, end to end)

The v7 value-SIGN bug history (same −107@50 Elo pattern reappearing in later champion
runs) demanded a bit-level audit of the q field from the next-mover's view through
emission, file format, mixing, and training. **The current pipeline is sign-correct.**
The −107@50 recurrences must have other causes (per docs/TODO.md's own note: 20-game
arenas have ±150–300 Elo CIs; the −107 margin sits inside combined noise).

### Method (now a permanent regression test: `tests/test_sample_q_sign.py`)

`value_lambda` enters ONLY the post-hoc finalization mix
(`value = (1−λ)·outcome + λ·q`); it never touches the RNG or the trajectory. So three
selfplay runs with identical seeds/threads/λ ∈ {0.0, 0.5, 1.0} produce **bit-identical
games** and differ only in the stored value field. Live run (60 games, 100 rollout
sims, seed 42, threads=1, `results` recorded in test):

| Check | Result |
|---|---|
| positions per λ | 1670 / 1670 / 1670, identical W-L-D 28-27-5 → identical trajectories |
| inputs & policies across λ | `array_equal` = True (pairing is bit-exact) |
| mixing formula `z@0.5 ≈ 0.5·z@0 + 0.5·q` | max abs err **0.0** (float-exact) |
| `z@λ=1.0` vs stored `q` field | identical (max err 0.0) |
| mean q where mover eventually WON | **+0.170** (n=752) |
| mean q where mover eventually LOST | **−0.165** (n=725) |
| mean q for draws | −0.007 (n=193) |
| corr(outcome_from_mover_view, q) | **+0.438** |

If the v7 sign bug were live, winners' q would be negative and the correlation
inverted. It is not. Win-in-1 shortcut samples carry q=+1 from the mover's view and
those movers really won (asserted in the test).

### Loss-side semantics (verified, `tests/test_train_targets.py`)

- `train.py step()`: policy loss `−Σ π·log_softmax` masked by `full_search` weights,
  normalized by `Σw` — masked-mean matches a manual computation to <1e-5; an
  all-value-only batch yields policy_loss exactly 0.0 while value MSE stays finite.
- Value head returns `[B]` (both MLP and ResNet `squeeze(-1)`), so
  `F.mse_loss(v, z)` can never silently broadcast `[B,1]` vs `[B]` into a B×B
  comparison — regression test added to keep it that way.
- `data.py read_samples`: v4 record stride verified field-by-field against a
  synthetic binary (input/policy/value/full/q). No misalignment.

### Fixes shipped for auditability (not bugs, but prerequisites for this audit)

- `data.py read_samples` now exposes the raw `q` array in the metadata dict for v4
  files (`None` for v2/v3). Without it, the mix formula could not be verified on real
  files from Python. All existing consumers use `.get(...)` — backward compatible.

---

## 2. Confirmed bug — `build_supports_nn` probed the wrong game

`orchestrator.py build_supports_nn(build)` hardcoded `--game connect4` (line 31) and
the single call site passed only the build dir. For any other game the probe answers
for connect4: it can return True for a build that lacks the trained game's NN glue
(poisoning every iteration's evaluator switch with load failures), or False for a
build that supports the actual game (silently freezing the run on rollout data
forever — an entire wasted training budget with no error).

**Fix**: `build_supports_nn(build, game)`, called as `build_supports_nn(a.build,
a.game)`; probe file moved to the approved `/tmp/opencode/` scratch dir.
Verified semantics against this build: `nn:` on a non-libtorch build → rc 3 →
False (correct); real champion runs (libtorch build) logged
`evaluator=nn-strict:...latest.ts`, i.e. probe→rc 0 there. Behavior on connect4 is
unchanged; behavior on every other game changes from "wrong" to "right".

Tests: `tests/test_orchestrator_plumbing.py::test_build_supports_nn_*` (command
capture: `--game` == the requested game, "connect4" absent; rc→bool mapping).

---

## 3. Confirmed bug — mirror augmentation silently corrupted non-symmetric games

`data.py maybe_mirror_batch(...)` had no notion of game: it mirrored the encoded
input and reversed the policy vector for ANY game whenever `selfplay.augment` was
set. For connect4 that's an exact symmetry; for any game whose encoding is not
left-right symmetric or whose policy index is not the mirrored column it writes
**silently corrupted policy targets** into the replay buffer — the exact class of
bug that lowers Elo with no crash and no log line. The repo's own template game
declares `kNumSymmetries = 1` (no symmetry hooks), so this was one config flag away
from firing.

**Fix**: registry `MIRROR_AUGMENTERS = {"connect4": mirror_augment_connect4}` +
`game=` kwarg, required. Unknown/missing game → `ValueError` naming the game, the
registry, and both remediations (register a verified handler or set
`selfplay.augment: 0`). Orchestrator call site now passes `game=a.game`. connect4
behavior is bit-identical (registry maps to the same function; deterministic-rng
test proves noop and mirror paths unchanged).

Tests: `tests/test_data_augment.py` (7 tests: connect4 unchanged / noop path /
missing-game raises / unknown-game raises / template excluded / actionable message).

---

## 4. steps/iter starvation — independently verified HEALTHY (no fix needed)

Agent C reported 10.6–11 steps/iter; I verified independently:

- **From real run logs** (fresh_positions / effective_steps fields):
  `c4_60min_vw2strict` 4812 iters mean **10.92** (min 8 / max 13),
  `c4_10min_v10_rebase2` mean **11.05**, `c4_10min_vw2` mean **11.06** —
  consistent with the intended ~11/iter, no starvation.
- **From the code**: the clamp denominator is `fresh_positions = len(train_z)`
  AFTER the held-out split and AFTER augmentation — the v9 starvation direction
  (raw count undercounts ~2× when augment is on → fewer allowed steps) cannot
  recur silently.

**Hardening shipped**: the inline clamp was extracted to a pure, documented function
`orchestrator.compute_effective_steps(steps_per_iter, fresh_positions, batch_size,
max_reuse_ratio, buffered_positions)` (behavior-identical — same int truncation,
same 1-step guard when the buffer can fill a batch). Unit tests pin: healthy regime
unclamped, clamp binding, doubling of denominator doubles allowed steps (v9 shape),
the 1-step anti-stall guard, and 0-steps-when-buffer-can't-fill. The guard logic is
now testable instead of buried in a 300-line `main()`.

Tests: `tests/test_orchestrator_plumbing.py::test_steps_*` (6 tests).

---

## 5. TorchScript export parity — verified, tests added

`export_torchscript` traces `net.eval()` on a zeros example. Verified for MLP and
small ResNet (BN blocks, global mean/max pool): loaded `.ts` matches eager logits
and value to <1e-5 on batch sizes different from the trace example (trace
generalization), and eval-mode BN stats are baked (train-mode dirty-stats net
exports identical to the same net in eval). No parity gap found — the C++
NNEvaluator serves from this export, so these are now permanent regression tests.

Tests: `tests/test_train_targets.py::test_torchscript_export_parity_both_archs`,
`test_export_freezes_eval_mode`.

---

## 6. Unconfirmed / observations (no action taken, out of my scope or no Elo evidence)

1. **`root_value()` is a visit-weighted mean over ALL root children** (agent-D
   territory; noted for the record): Dirichlet-forced and cheap visits of bad moves
   drag the stored q toward 0 vs the best-child value. With λ=0.1 the effect on the
   target is bounded (~0.1·spread). If the −107@50 pattern is ever chased further,
   an A/B of `q = best_child_q()` vs the current mean (flag-gated in selfplay) is a
   cheap experiment. 10-min check: rerun the λ-sweep with `resign off` and compare
   `corr(q, outcome)` for mean-q vs best-child-q variants.
2. **`read_samples` loses game boundaries** (samples concatenated per record). Any
   future per-game analysis (e.g. resign false-positive calibration in Python)
   needs a positions-per-game side channel from the C++ tool. Feature gap, not a bug.
3. **`q` exposure is v4-only by design**; v2/v3 files yield `q=None` — callers must
   handle None if they mix old data (orchestrator never reads q, so no consumer
   changed).

## Verification

```
python3 -m pytest python/tests -q   →  83 passed, 0 failed, 0 skipped
orchestrator --dry-run (connect4, v2 config) → 2 iters, probe/train/export/arena OK
```

Files touched (all inside the narrowed scope):
- `python/gai/data.py` (q exposure; augmenter registry + loud failure)
- `python/gai/orchestrator.py` (probe takes game; augment call passes game;
  compute_effective_steps extraction)
- `python/tests/test_data_augment.py` (new, 7 tests)
- `python/tests/test_orchestrator_plumbing.py` (new, 8 tests)
- `python/tests/test_sample_q_sign.py` (new, 3 tests — incl. live λ-sweep)
- `python/tests/test_train_targets.py` (new, 5 tests)
