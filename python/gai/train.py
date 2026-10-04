"""Training step + checkpointing + TorchScript export."""
import time, torch, torch.nn.functional as F
from .model import build

def make_trainer(net_cfg, shape, lr=1e-3, wd=1e-4, amp=True, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = build(net_cfg, shape["planes"], shape["h"], shape["w"], shape["actions"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.amp.GradScaler(enabled=(amp and device == "cuda"))
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
            loss = lp + lv
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        return dict(loss=loss.item(), policy_loss=lp.item(), value_loss=lv.item())
    return net, opt, step

def save_checkpoint(net, path, extra=None):
    torch.save({"state": net.state_dict(), **(extra or {})}, path)

def load_checkpoint(net, path, device="cpu"):
    sd = torch.load(path, map_location=device)
    net.load_state_dict(sd["state"] if "state" in sd else sd)

def export_torchscript(net, shape, path):
    """For the C++ NNEvaluator (libtorch). Input [B,planes,h,w] -> (policy_logits[B,A], value[B])."""
    net.eval(); ex = torch.zeros(1, shape["planes"], shape["h"], shape["w"], device=next(net.parameters()).device)
    torch.jit.trace(net, ex).save(path)
