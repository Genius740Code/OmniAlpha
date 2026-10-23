"""Training-target semantics: masked policy loss + value sign safety.

* policy loss is masked to full_search samples only (KataGo playout-cap
  semantics; cheap/shortcut samples are value-only);
* the value head returns [B] so mse_loss(v, z) can never silently broadcast
  [B,1] vs [B] into a BxB comparison (which would collapse the value head
  toward the batch mean);
* TorchScript export is output-identical to the eager model (the C++
  NNEvaluator serves from the export, so any parity gap poisons self-play).
"""
import numpy as np
import torch
import torch.nn.functional as F

from gai.model import build
from gai.train import make_trainer, export_torchscript

SHAPE = dict(planes=1, h=2, w=2, actions=3)
CFG = {"type": "mlp", "hidden": 8, "layers": 1}


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 1, 2, 2)).astype(np.float32)
    pi = rng.random((n, 3)).astype(np.float32)
    pi /= pi.sum(-1, keepdims=True)
    z = rng.random(n).astype(np.float32) * 2 - 1
    return x, pi, z


def test_policy_loss_is_masked_to_full_search_rows():
    net, opt, step, set_lr, ema = make_trainer(CFG, SHAPE, value_weight=0.0, ema_decay=0.0)
    x, pi, z = _batch(4)
    full = np.array([1.0, 0.0, 1.0, 0.0], np.float32)
    net.eval()
    with torch.no_grad():
        logits, _ = net(torch.as_tensor(x))
        per = -(torch.as_tensor(pi) * F.log_softmax(logits.float(), -1)).sum(-1)
        expected = (per * torch.as_tensor(full)).sum() / torch.as_tensor(full).sum()
    out = step(x, pi, z, full)  # loss values are computed before the update
    assert abs(out["policy_loss"] - expected.item()) < 1e-5


def test_all_value_only_batch_has_zero_policy_loss():
    net, opt, step, set_lr, ema = make_trainer(CFG, SHAPE, ema_decay=0.0)
    x, pi, z = _batch(4)
    full = np.zeros(4, np.float32)
    out = step(x, pi, z, full)
    assert out["policy_loss"] == 0.0  # 0-weight numerator / clamped denominator
    assert np.isfinite(out["value_loss"])  # value head still learns from z


def test_value_head_shapes_prevent_broadcast_trap():
    for cfg in (CFG, {"type": "resnet", "channels": 4, "blocks": 1}):
        net = build(cfg, 1, 2, 2, 3)
        x, _, z = _batch(5)
        logits, v = net(torch.as_tensor(x))
        assert v.shape == (5,)          # [B], not [B,1]
        assert logits.shape == (5, 3)
        # the exact expression train.py uses must not broadcast
        assert F.mse_loss(v.float(), torch.as_tensor(z)).ndim == 0


def test_torchscript_export_parity_both_archs(tmp_path):
    for cfg in (CFG, {"type": "resnet", "channels": 4, "blocks": 1}):
        torch.manual_seed(0)
        net = build(cfg, 1, 2, 2, 3)
        ts = str(tmp_path / f"{cfg['type']}.ts")
        export_torchscript(net, SHAPE, ts)
        loaded = torch.jit.load(ts)
        loaded.eval()
        x, _, _ = _batch(7, seed=3)  # batch != export example (1)
        with torch.no_grad():
            lg_e, v_e = net(torch.as_tensor(x))
            lg_t, v_t = loaded(torch.as_tensor(x))
        assert torch.allclose(lg_e, lg_t, atol=1e-5), cfg
        assert torch.allclose(v_e, v_t, atol=1e-5), cfg


def test_export_freezes_eval_mode(tmp_path):
    # train-mode BN/dropout behaviour must not leak into the served module
    cfg = {"type": "resnet", "channels": 4, "blocks": 1}
    net = build(cfg, 1, 2, 2, 3).train()
    # dirty the BN running stats so eval-vs-train differ
    for _ in range(3):
        net(torch.as_tensor(_batch(8, seed=1)[0]))
    ts = str(tmp_path / "bn.ts")
    export_torchscript(net, SHAPE, ts)
    net.eval()
    loaded = torch.jit.load(ts)
    x, _, _ = _batch(4, seed=2)
    with torch.no_grad():
        lg_e, v_e = net(torch.as_tensor(x))
        lg_t, v_t = loaded(torch.as_tensor(x))
    assert torch.allclose(lg_e, lg_t, atol=1e-4)
    assert torch.allclose(v_e, v_t, atol=1e-4)
