#!/usr/bin/env python3
"""Margin-scale sweep: value-target recalibration on a shared trunk.

The value target is tanh(final margin / SCALE). 1500 was the original choice;
smaller scales (600, 900) spread close-game targets over more of the tanh's
range, which should sharpen close-game predictions. Sign accuracy is
scale-invariant as a metric (sign(tanh(m/s)) == sign(m) for any s > 0), so
what the sweep compares is which TRAINING scale optimizes best.

Protocol: from a trained checkpoint, freeze the trunk + policy head and
fine-tune only the value head at each candidate scale over a small balanced
sample (the trunk features are scale-agnostic; this isolates the head's
calibration cheaply). The winner can then be exported directly, since only the
value head moved.

  python3 sweep.py --ckpt runs/fast_v1/best.pt
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset
from model import TrainNet
from train import build_val, quick_val


def finetune_value_head(base, recs, pool, margin_pool, scale, steps, batch,
                        lr, np_rng, seed):
    """Returns a copy of `base` with only the value head retrained at `scale`."""
    torch.manual_seed(seed)
    m = copy.deepcopy(base)
    m.train()
    for name, p in m.named_parameters():
        p.requires_grad = name.startswith(("v.", "v_bn.", "v_fc."))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    weights = dataset.balance_weights(margin_pool)
    weights = weights / weights.sum()
    idx_all = np_rng.choice(pool, size=steps * batch, p=weights)
    last = None
    for b in range(steps):
        idx = idx_all[b * batch:(b + 1) * batch]
        x, v_t, _ = dataset.make_batch(recs, idx, scale, np_rng.integers(0, 4, idx.size))
        v, _ = m(torch.from_numpy(x).contiguous(memory_format=torch.channels_last), policy=False)
        loss = F.mse_loss(v, torch.from_numpy(v_t))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss)
    m.eval()
    return m, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="/tmp/opencode/train2")
    ap.add_argument("--scales", type=float, nargs="+", default=[600.0, 900.0, 1500.0])
    ap.add_argument("--steps", type=int, default=60, help="value-head fine-tune steps per scale")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(2)
    recs, is_val = dataset.load_split(args.out)
    ply = dataset.column(recs, "ply")
    margin = dataset.column(recs, "margin")
    pool = np.flatnonzero(~is_val & (ply >= 6))
    margin_pool = margin[pool]

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base = TrainNet()
    base.load_state_dict(ckpt["model"])
    base.eval()
    base_scale = ckpt["meta"].get("scale", 1500.0)

    x_val_full, margin_val_full, _ = build_val(recs, is_val, stride=1)
    x_val_q, margin_val_q, _ = build_val(recs, is_val, stride=5)

    results = {"base": {"scale": base_scale,
                        **quick_val(base, x_val_q, margin_val_q, base_scale)}}
    print(json.dumps({"base": results["base"]}), flush=True)
    best_scale, best_key = None, -1.0
    for scale in args.scales:
        np_rng = np.random.default_rng(args.seed)
        m, last = finetune_value_head(base, recs, pool, margin_pool, scale,
                                      args.steps, args.batch, args.lr, np_rng, args.seed)
        q = quick_val(m, x_val_q, margin_val_q, scale)
        full = quick_val(m, x_val_full, margin_val_full, scale)
        key = 0.5 * full["sign"] + 0.5 * full["close_sign"]
        rec = {"scale": scale, "train_mse_last": round(last, 5), "quick": q, "full": full}
        results[f"scale_{scale:g}"] = rec
        print(json.dumps(rec), flush=True)
        if key > best_key:
            best_key, best_scale = key, scale
            os.makedirs(os.path.join(args.out, "runs", "sweep"), exist_ok=True)
            torch.save({"model": m.state_dict(),
                        "meta": {**ckpt["meta"], "scale": scale,
                                 "swept_from": os.path.basename(args.ckpt)}},
                       os.path.join(args.out, "runs", "sweep", "best_sweep.pt"))
    print(json.dumps({"best_scale": best_scale, "key": round(best_key, 4)}))
    with open(os.path.join(args.out, "runs", "sweep", "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
