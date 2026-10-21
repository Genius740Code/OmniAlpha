#!/usr/bin/env python3
"""Evaluates a net on the validation split (torch), with close-game breakdown.

  python3 bench.py --weights ../../src/weights.rs            # a weights.rs file
  python3 bench.py --ckpt runs/foo/best.pt                   # a training checkpoint

Reports, in tanh(margin/scale) space: mse, sign accuracy and correlation,
overall and on the close-game subset (|final margin| <= 300), plus unmasked
policy top-1 / top-8. The overall numbers should track `nnbench --val` on the
same games (the Rust harness additionally masks policy by legality).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
from model import NUM_MOVES, FoldedNet, TrainNet, fold, folded_model_from_flat, load_folded_from_rs


def evaluate(net: FoldedNet, x, margin, move, scale, batch=512, close=dataset.CLOSE, stride=1):
    """Value metrics overall + close subset; policy top-1/top-8 (unmasked)."""
    n = x.shape[0]
    preds = np.empty(n, np.float32)
    with torch.no_grad():
        for i in range(0, n, batch):
            v, _ = net(torch.from_numpy(x[i:i + batch]), policy=False)
            preds[i:i + batch] = v.numpy()
    target = np.tanh(margin.astype(np.float64) / scale).astype(np.float32)
    am = np.abs(margin)

    def stats(sel):
        p, t = preds[sel], target[sel]
        mse = float(np.mean((p - t) ** 2))
        sign = float(np.mean((p >= 0) == (t >= 0)))
        c = np.corrcoef(p, t)[0, 1] if sel.sum() > 2 else float("nan")
        return {"n": int(sel.sum()), "mse": mse, "sign": sign, "corr": float(c)}

    out = {"overall": stats(np.ones(n, bool)), f"close_{int(close)}": stats(am <= close),
           "wide": stats(am > close)}
    return out, preds


def policy_metrics(net: FoldedNet, x, move, batch=256, maxn=20000):
    idx = np.linspace(0, x.shape[0] - 1, min(maxn, x.shape[0])).astype(np.int64)
    top1 = top8 = 0
    with torch.no_grad():
        for i in range(0, idx.size, batch):
            sel = idx[i:i + batch]
            _, p = net(torch.from_numpy(x[sel]), policy=True)
            top = torch.topk(p, 8, dim=1).indices.numpy()
            top1 += (top[:, 0] == move[sel]).sum()
            top8 += (top == move[sel][:, None]).any(1).sum()
    return {"n": int(idx.size), "top1": float(top1 / idx.size), "top8": float(top8 / idx.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="a weights.rs file to evaluate (folded)")
    ap.add_argument("--ckpt", help="a training checkpoint (TrainNet, will be folded)")
    ap.add_argument("--scale", type=float, default=1500.0)
    ap.add_argument("--out", default="/tmp/opencode/train2")
    ap.add_argument("--max-policy", type=int, default=20000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.set_num_threads(2)
    recs, is_val = dataset.load_split(args.out)
    ply = dataset.column(recs, "ply")
    sel = is_val & (ply >= 6)
    planes, scalars, move, margin, _ = dataset.unpack(recs[sel])
    n = int(sel.sum())
    x = np.zeros((n, 14, dataset.BOARD, dataset.BOARD), np.float32)
    x[:, :9] = planes.reshape(n, 9, dataset.BOARD, dataset.BOARD)
    x[:, 9:] = scalars[:, :, None, None]
    margin, move = margin, move

    if args.weights:
        net = load_folded_from_rs(args.weights)
        name = os.path.basename(args.weights)
    else:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        m = TrainNet()
        m.load_state_dict(sd["model"] if "model" in sd else sd)
        net = folded_model_from_flat(fold(m))
        name = os.path.basename(args.ckpt)
    out, _ = evaluate(net, x, margin, move, args.scale)
    out["policy"] = policy_metrics(net, x, move, maxn=args.max_policy)
    out["tag"] = args.tag or name
    out["scale"] = args.scale
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
