"""Training step + checkpointing + TorchScript export."""
import math, time, torch, torch.nn.functional as F
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
                 ema_decay=0.0):
    """B6: value_weight scales the value MSE vs policy CE; grad_clip>0 enables
    global-norm clipping; optimizer in {"adamw","sgd"} (sgd uses momentum).

    ema_decay>0 maintains an exponential moving average of the parameters
    (buffers such as BatchNorm running stats are copied, not averaged) updated
    after every optimizer step: ema = decay*ema + (1-decay)*param. Returns a
    5-tuple (net, opt, step, set_lr, ema) where ema is None when disabled and
    otherwise dict(name -> tensor). Export it with ema_export().
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
        ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
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
    torch.save({"state": net.state_dict(), **(extra or {})}, path)

def load_checkpoint(net, path, device="cpu"):
    sd = torch.load(path, map_location=device)
    net.load_state_dict(sd["state"] if "state" in sd else sd)

def export_torchscript(net, shape, path):
    """For the C++ NNEvaluator (libtorch). Input [B,planes,h,w] -> (policy_logits[B,A], value[B])."""
    net.eval(); ex = torch.zeros(1, shape["planes"], shape["h"], shape["w"], device=next(net.parameters()).device)
    torch.jit.trace(net, ex).save(path)
