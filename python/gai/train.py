"""Training step + checkpointing + TorchScript export."""
import math, os, time, torch, torch.nn.functional as F
from .model import build

def setup_torch_threads(n):
    """Cap intra-op (and inter-op) threads for CPU training. n<=0 is a no-op.

    On the 2-core competition box torch defaults to 2 intra-op threads, which is
    usually right, but small MLP batches often run FASTER single-threaded (less
    sync overhead) — tune with training.num_threads. Returns the thread count in
    effect after the call (from torch.get_num_threads()).
    """
    n = int(n or 0)
    if n <= 0:
        return torch.get_num_threads()
    torch.set_num_threads(n)
    try:  # interop pool can only be configured before parallel work starts
        torch.set_num_interop_threads(1)
    except (RuntimeError, ValueError):
        pass
    try:  # denormals can wreck CPU matmul latency; flushing them is free
        torch.set_flush_denormal(True)
    except (RuntimeError, ValueError, AttributeError):
        pass
    return torch.get_num_threads()

def cosine_lr(base_lr, frac, warmup_frac=0.03, min_lr_frac=0.02):
    """Warmup + cosine decay floor. Pure function (unit-tested).

    frac in [0,1] is elapsed/training-window; below warmup_frac LR ramps linearly
    0->base, then decays cosine to base*min_lr_frac at frac=1. Matches the inline
    schedule the orchestrator used before (min_lr_frac default 0.02).
    """
    warmup_frac = max(0.0, float(warmup_frac))
    min_lr_frac = float(min_lr_frac)
    if warmup_frac > 0 and frac < warmup_frac:
        return base_lr * (frac / warmup_frac)
    return base_lr * (min_lr_frac + (1.0 - min_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * frac)))

def make_trainer(net_cfg, shape, lr=1e-3, wd=1e-4, amp=True, device=None,
                 value_weight=1.0, grad_clip=0.0, optimizer="adamw", momentum=0.9,
                 ema_decay=0.0, ema_init=None):
    """B6: value_weight scales the value MSE vs policy CE; grad_clip>0 enables
    global-norm clipping; optimizer in {"adamw","sgd"} (sgd uses momentum).

    ema_decay>0 maintains an exponential moving average of the parameters
    (buffers such as BatchNorm running stats are copied, not averaged) updated
    after every optimizer step: ema = decay*ema + (1-decay)*param. Returns a
    5-tuple (net, opt, step, set_lr, ema) where ema is None when disabled and
    otherwise dict(name -> tensor). Export it with ema_export().

    ema_init (from load_train_state) seeds the EMA dict on resume; keys missing
    from it fall back to the freshly initialised net, so old states stay loadable.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = build(net_cfg, shape["planes"], shape["h"], shape["w"], shape["actions"]).to(device)
    if optimizer == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler(enabled=(amp and device == "cuda"))
    ema_decay = float(ema_decay)
    ema = None
    param_names = {n for n, _ in net.named_parameters()}
    if 0.0 < ema_decay < 1.0:
        if ema_init is None:
            ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
        else:  # resume: keep the saved average, fill any new keys from the net
            base = net.state_dict()
            ema = {k: (ema_init[k].detach().to(device=device, dtype=torch.float32).clone()
                       if k in ema_init else v.detach().clone().float())
                   for k, v in base.items()}
        def ema_update():
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if k in param_names:
                        ema[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                    else:  # buffers (BN running_mean/var, num_batches) track exactly
                        ema[k].copy_(v)
    else:
        def ema_update():
            pass
    def set_lr(new_lr):
        for pg in opt.param_groups: pg["lr"] = new_lr
    def step(x, pi, z, full=None):
        net.train(); x, pi, z = [torch.as_tensor(a, device=device) for a in (x, pi, z)]
        if full is None:
            w = torch.ones_like(z)
        else:
            w = torch.as_tensor(full, device=device, dtype=torch.float32)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(amp and device == "cuda")):
            logits, v = net(x)
            per = -(pi * F.log_softmax(logits.float(), -1)).sum(-1)
            lp = (per * w).sum() / w.sum().clamp_min(1e-6)
            lv = F.mse_loss(v.float(), z)
            loss = lp + value_weight * lv
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        scaler.step(opt); scaler.update()
        ema_update()
        # return tensors; orchestrator extracts .item() at iter boundaries to avoid per-step overhead
        return dict(loss=loss.item(), policy_loss=lp.item(), value_loss=lv.item())
    return net, opt, step, set_lr, ema

def ema_export(ema, net, shape, path_pt, path_ts):
    """Write the EMA weights as a checkpoint + TorchScript pair (best-effort ts).

    Uses a throwaway net instance with the EMA state loaded; the live training net
    is untouched. Raises on failure — callers decide fatal vs warn.
    """
    import copy as _copy
    enet = _copy.deepcopy(net)
    enet.load_state_dict({k: v.to(next(net.parameters()).device) for k, v in ema.items()})
    save_checkpoint(enet, path_pt)
    export_torchscript(enet, shape, path_ts)

def save_checkpoint(net, path, extra=None):
    # Atomic rename: a reader loading the checkpoint mid-write (endgame tournament,
    # post-hoc scripts) must never see a truncated file.
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save({"state": net.state_dict(), **(extra or {})}, tmp)
    os.replace(tmp, path)

def load_checkpoint(net, path, device="cpu"):
    sd = torch.load(path, map_location=device, weights_only=False)  # trusted local artifact
    net.load_state_dict(sd["state"] if "state" in sd else sd)

def save_train_state(path, net, opt, ema, it, elapsed_s, extra=None):
    """Atomic resume state: weights + optimizer + EMA + iteration/cumulative clock.

    Separate from checkpoint_*.pt (which stay inference artifacts): only the
    orchestrator reads this, and every save is an os.replace so a crash during a
    10-hour run can never leave a half-written state file. `extra` carries the
    shape so --resume can log/validate before rebuilding the net.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    payload = {"format": 1, "model": net.state_dict(),
               "optimizer": opt.state_dict() if opt is not None else None,
               "ema": ema, "iter": int(it), "elapsed_s": float(elapsed_s)}
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    os.replace(tmp, path)

def load_train_state(path, net=None, opt=None, device="cpu", load_weights=True):
    """Read a save_train_state file. Returns metadata even when only peeking.

    net/opt are restored in place when given (weights/optimizer moments). EMA is
    returned for make_trainer(ema_init=...), which installs it inside the closure
    that owns the update step. `load_weights=False` peeks at metadata only.
    """
    sd = torch.load(path, map_location=device, weights_only=False)  # trusted local artifact
    if net is not None and load_weights:
        net.load_state_dict(sd["model"] if "model" in sd else sd["state"])
    if opt is not None and sd.get("optimizer") is not None:
        opt.load_state_dict(sd["optimizer"])
    return {"iter": int(sd.get("iter", 0)),
            "elapsed_s": float(sd.get("elapsed_s", 0.0)),
            "ema": sd.get("ema"),
            "shape": sd.get("shape"),
            "format": int(sd.get("format", 0))}

def export_torchscript(net, shape, path):
    """For the C++ NNEvaluator (libtorch). Input [B,planes,h,w] -> (policy_logits[B,A], value[B]).

    Atomic: writes to a private temp file in the same directory, then os.replace(),
    so a serving-side hot reload (play --reload / NNEvaluator lazy load) can never
    read a half-written .ts while training exports.
    """
    net.eval(); ex = torch.zeros(1, shape["planes"], shape["h"], shape["w"], device=next(net.parameters()).device)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.jit.trace(net, ex).save(tmp)
    os.replace(tmp, path)
