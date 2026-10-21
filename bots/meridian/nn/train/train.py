#!/usr/bin/env python3
"""Supervised trainer for the Meridian value/policy net.

Two profiles:
  baseline: the old recipe reconstructed — every position (no curriculum),
    uniform sampling, no augmentation, one pass over the whole corpus
    per "epoch", value scale fixed at 1500.
  fast (default): the new recipe — curriculum (skip the first 6 plies),
    margin-balanced sampling (close games ~3x), 4-fold symmetry
    augmentation, cosine LR, best-epoch selection on a value+close metric.

Both train the same architecture (model.TrainNet) with the same optimizer, so
the profiles isolate exactly the sample-efficiency changes.

  python3 train.py --profile fast    --epochs 6 --samples-per-epoch 160000
  python3 train.py --profile baseline --epochs 1 --samples-per-epoch 0   # 0 = whole corpus
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import dataset
from model import TrainNet


def build_val(recs, is_val, stride=1):
    """Materializes the validation tensors once (no augmentation)."""
    ply = dataset.column(recs, "ply")
    sel = np.flatnonzero(is_val & (ply >= 6))
    if stride > 1:
        sel = sel[::stride]
    planes, scalars, move, margin, _ = dataset.unpack(recs[sel])
    n = sel.size
    x = np.zeros((n, 14, dataset.BOARD, dataset.BOARD), np.float32)
    x[:, :9] = planes.reshape(n, 9, dataset.BOARD, dataset.BOARD)
    x[:, 9:] = scalars[:, :, None, None]
    return x, margin, move


@torch.no_grad()
def quick_val(model, x_val, margin_val, scale, batch=512, close=dataset.CLOSE):
    """Value-only metrics on the validation tensors."""
    model.eval()
    preds = np.empty(x_val.shape[0], np.float32)
    for i in range(0, x_val.shape[0], batch):
        v, _ = model(torch.from_numpy(x_val[i:i + batch]), policy=False)
        preds[i:i + batch] = v.numpy()
    model.train()
    target = np.tanh(margin_val.astype(np.float64) / scale).astype(np.float32)
    am = np.abs(margin_val)
    sign = float(np.mean((preds >= 0) == (target >= 0)))
    mse = float(np.mean((preds - target) ** 2))
    sel = am <= close
    close_sign = float(np.mean((preds[sel] >= 0) == (target[sel] >= 0)))
    return {"sign": sign, "mse": mse, f"close_sign": close_sign,
            "n": int(x_val.shape[0]), "n_close": int(sel.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/opencode/train2")
    ap.add_argument("--run", default="run")
    ap.add_argument("--profile", choices=["fast", "baseline"], default="fast")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=160000,
                    help="0 = a full pass over the corpus (baseline)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--scale", type=float, default=1500.0)
    ap.add_argument("--val-stride", type=int, default=5)
    ap.add_argument("--curriculum", type=int, default=6)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--probe", type=int, default=0, help="run N probe steps and exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.set_num_threads(2)
    run_dir = os.path.join(args.out, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)

    recs, is_val = dataset.load_split(args.out)
    ply = dataset.column(recs, "ply")
    margin = dataset.column(recs, "margin")
    train_mask = ~is_val
    if args.profile == "fast":
        train_mask &= ply >= args.curriculum
    train_pool = np.flatnonzero(train_mask)
    print(f"corpus: {recs.shape[0]} positions, train pool {train_pool.size}, "
          f"val {int((is_val & (ply >= 6)).sum())} (ply>={args.curriculum} train filter)")

    if args.profile == "baseline" or args.no_balance:
        weights = None
        samples_per_epoch = (train_pool.size if args.samples_per_epoch == 0
                             else args.samples_per_epoch)
    else:
        weights = dataset.balance_weights(margin[train_pool])
        weights = weights / weights.sum()
        samples_per_epoch = args.samples_per_epoch or train_pool.size

    x_val, margin_val, _ = build_val(recs, is_val, stride=args.val_stride)
    print(f"val tensors: {x_val.shape[0]} positions (stride {args.val_stride})")

    model = TrainNet()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = (samples_per_epoch + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    warmup = min(200, total_steps // 10)

    def lr_at(step):
        if step < warmup:
            return args.lr * (0.1 + 0.9 * step / max(warmup, 1))
        t = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t)))

    aug = not (args.profile == "baseline" or args.no_aug)
    t_start = time.time()
    step = 0
    best = {"score": -1.0}
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    for epoch in range(args.epochs):
        if weights is None:
            idx_epoch = (np_rng.permutation(train_pool)[:samples_per_epoch]
                         if samples_per_epoch < train_pool.size else np_rng.permutation(train_pool))
        else:
            idx_epoch = np_rng.choice(train_pool, size=samples_per_epoch, p=weights)
        model.train()
        t_epoch = time.time()
        t_collate = 0.0
        losses = []
        for b in range(steps_per_epoch):
            idx = idx_epoch[b * args.batch:(b + 1) * args.batch]
            if idx.size == 0:
                break
            t0 = time.time()
            x, v_t, m_t = dataset.make_batch(
                recs, idx, args.scale,
                np_rng.integers(0, 4, idx.size) if aug else None)
            xt = torch.from_numpy(x)
            if args.channels_last:
                xt = xt.contiguous(memory_format=torch.channels_last)
            t_collate += time.time() - t0

            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            v, p = model(xt)
            loss = F.mse_loss(v, torch.from_numpy(v_t)) + F.cross_entropy(p, torch.from_numpy(m_t))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            step += 1
            if args.probe and step >= args.probe:
                dt = time.time() - t_start
                print(json.dumps({
                    "probe": True, "steps": step, "samples": step * args.batch,
                    "seconds": round(dt, 2),
                    "samples_per_min": round(step * args.batch / dt * 60),
                    "collate_frac": round(t_collate / dt, 3),
                    "batch": args.batch, "channels_last": args.channels_last,
                }))
                return

        dt = time.time() - t_epoch
        m = quick_val(model, x_val, margin_val, args.scale)
        score = 0.5 * m["sign"] + 0.5 * m["close_sign"]
        rec = {"epoch": epoch, "steps": steps_per_epoch, "minutes": round(dt / 60, 2),
               "samples_per_min": round(samples_per_epoch / dt * 60),
               "collate_frac": round(t_collate / dt, 3),
               "loss": round(float(np.mean(losses)), 4), "score": round(score, 4),
               "lr": lr_at(step), **{k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in m.items()}}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, **m}
            torch.save({"model": model.state_dict(), "meta": {**rec, "scale": args.scale,
                       "profile": args.profile, "aug": aug, "balanced": weights is not None,
                       "curriculum": args.curriculum if args.profile == "fast" else 0,
                       "batch": args.batch, "lr": args.lr}},
                       os.path.join(run_dir, "best.pt"))

    print(json.dumps({"best": best, "total_minutes": round((time.time() - t_start) / 60, 2)}))
    log.write(json.dumps({"best": best}) + "\n")
    log.close()


if __name__ == "__main__":
    main()
