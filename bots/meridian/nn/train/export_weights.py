#!/usr/bin/env python3
"""Exports a training checkpoint to bots/meridian/nn/src/weights.rs and the
torch-parity artifacts `nnbench --check` needs.

  python3 export_weights.py --ckpt runs/fast/best.pt
  python3 export_weights.py --ckpt ... --nnbench-artifacts

Writes:
  bots/meridian/nn/src/weights.rs          (same 26-tensor LAYOUT, new bytes)
  <out>/nn-data/out/expected_dense.bin      torch values on the dense val samples
  and (with --nnbench-artifacts) copies val_games.json + val_dense_head.bin to
  the fixed paths nnbench looks in: /tmp/opencode/nn-data/{out,out3}/.

Also runs two self-tests before writing anything:
  - fold parity: TrainNet vs its folded self on random inputs (max |diff|),
  - decode parity: parsing the generated file back gives the same bytes.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
from model import (FoldedNet, LAYOUT, TrainNet, export_f32, fold,
                   folded_model_from_flat, folded_state_dict_from_bytes,
                   parse_weights_rs, write_weights_rs)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
WEIGHTS_RS = os.path.join(REPO, "bots/meridian/nn/src/weights.rs")


def fold_parity_check(model: TrainNet, recs) -> float:
    """TrainNet vs its folded self on real positions (in-distribution)."""
    model.eval()
    folded = folded_model_from_flat(fold(model))
    idx = np.linspace(0, recs.shape[0] - 1, 512).astype(np.int64)
    x, _, _ = dataset.make_batch(recs, idx, 1500.0, None)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        v1, p1 = model(xt)
        v2, p2 = folded(xt)
    err_v = (v1 - v2).abs().max().item()
    err_p = (p1 - p2).abs().max().item()
    assert err_v < 2e-5, f"fold value parity {err_v}"
    assert err_p < 1e-4, f"fold logit parity {err_p} (logits are O(30); relative ~1e-6)"
    return max(err_v, err_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="/tmp/opencode/train2")
    ap.add_argument("--weights-rs", default=WEIGHTS_RS)
    ap.add_argument("--nnbench-artifacts", action="store_true",
                    help="copy artifacts into /tmp/opencode/nn-data/ for nnbench --check/--val")
    ap.add_argument("--header", default="", help="extra header note")
    args = ap.parse_args()

    torch.set_num_threads(2)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    model = TrainNet()
    model.load_state_dict(ckpt["model"])
    model.eval()

    recs, _ = dataset.load_split(args.out)
    err = fold_parity_check(model, recs)
    flat = export_f32(fold(model))
    assert flat.size == 101_985 and flat.dtype == np.dtype("<f4")

    header = [
        f"val: sign {meta.get('sign', 0):.4f}, close sign {meta.get('close_sign', 0):.4f}, "
        f"mse {meta.get('mse', 0):.4f}, epoch {meta.get('epoch', '?')}, "
        f"{meta.get('minutes', 0)} min/epoch, scale {meta.get('scale', 1500):.0f}, "
        f"aug {int(meta.get('aug', 0))}, balanced {int(meta.get('balanced', 0))}, "
        f"curriculum {meta.get('curriculum', 0)}, profile {meta.get('profile', '?')}",
        args.header,
    ]
    write_weights_rs(args.weights_rs, flat, [h for h in header if h])

    # Decode-parity self-test: the file we just wrote must parse back exactly.
    reparsed = np.concatenate(parse_weights_rs(args.weights_rs))
    assert np.array_equal(reparsed, flat), "weights.rs round-trip mismatch"
    print(f"export ok: {flat.size} floats -> {args.weights_rs} "
          f"({os.path.getsize(args.weights_rs) / 1e6:.2f} MB file), fold parity {err:.2e}")

    # torch parity artifacts for nnbench --check
    nn_data = os.path.join(args.out, "nn-data")
    dense_path = os.path.join(nn_data, "out", "val_dense_head.bin")
    dense = np.fromfile(dense_path, dtype="<f4")
    per = 14 * 361 + 1
    n = dense.size // per
    x = torch.from_numpy(dense.reshape(n, per)[:, :per - 1].copy().reshape(n, 14, 19, 19))
    net = folded_model_from_flat(fold(model))
    values = np.empty(n, "<f4")
    with torch.no_grad():
        for i in range(0, n, 256):
            v, _ = net(x[i:i + 256], policy=False)
            values[i:i + 256] = v.numpy()
    expected = os.path.join(nn_data, "out", "expected_dense.bin")
    values.tofile(expected)
    print(f"expected_dense.bin: {n} torch values (max |v| {np.abs(values).max():.4f})")

    if args.nnbench_artifacts:
        fixed = "/tmp/opencode/nn-data"
        os.makedirs(os.path.join(fixed, "out"), exist_ok=True)
        os.makedirs(os.path.join(fixed, "out3"), exist_ok=True)
        shutil.copy(expected, os.path.join(fixed, "out", "expected_dense.bin"))
        shutil.copy(dense_path, os.path.join(fixed, "out", "val_dense_head.bin"))
        shutil.copy(os.path.join(nn_data, "out3", "val_games.json"),
                    os.path.join(fixed, "out3", "val_games.json"))
        print(f"artifacts copied to {fixed} (nnbench --check / --val defaults)")


if __name__ == "__main__":
    main()
