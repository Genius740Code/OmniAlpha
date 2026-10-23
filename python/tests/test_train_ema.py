"""Round C train-side: EMA weights, LR schedule helper, torch thread cap (train.py)."""
import numpy as np
import torch
from gai.train import make_trainer, ema_export, cosine_lr, load_checkpoint, setup_torch_threads

SHAPE = dict(planes=1, h=2, w=2, actions=3)
CFG = {"type": "mlp", "hidden": 8, "layers": 1}


def _batch(n=4):
    return (np.random.rand(n, 1, 2, 2).astype(np.float32),
            np.full((n, 3), 1 / 3, np.float32),
            np.zeros(n, np.float32))


def test_ema_disabled_returns_none():
    net, opt, step, set_lr, ema = make_trainer(CFG, SHAPE, ema_decay=0.0)
    assert ema is None


def test_ema_after_one_step_is_blend():
    net, opt, step, set_lr, ema = make_trainer(CFG, SHAPE, ema_decay=0.5)
    before = {k: v.clone() for k, v in ema.items()}
    step(*_batch())
    w0 = before["pol.weight"]
    w1 = net.state_dict()["pol.weight"]
    assert torch.allclose(ema["pol.weight"], 0.5 * w0 + 0.5 * w1, atol=1e-6)


def test_ema_decay_blends_repeatedly():
    net, opt, step, set_lr, ema = make_trainer(CFG, SHAPE, ema_decay=0.9)
    before = {k: v.clone() for k, v in ema.items()}
    for _ in range(3):
        step(*_batch())
    w1 = net.state_dict()["pol.weight"]
    # three steps of decay 0.9 from `before` with the FINAL weights only approximates
    # (weights move each step); just check the EMA moved less than the raw weights.
    assert (ema["pol.weight"] - before["pol.weight"]).norm() < (w1 - before["pol.weight"]).norm()


def test_ema_export_roundtrip():
    net, opt, step, set_lr, ema = make_trainer(CFG, SHAPE, ema_decay=0.7)
    step(*_batch())
    ema_export(ema, net, SHAPE, "/tmp/gai_ema_test.pt", "/tmp/gai_ema_test.ts")
    n3, _, _, _, _ = make_trainer(CFG, SHAPE, ema_decay=0.7)
    load_checkpoint(n3, "/tmp/gai_ema_test.pt")
    for k, v in ema.items():
        assert torch.allclose(n3.state_dict()[k], v)


def test_resnet_buffers_copied_not_averaged():
    cfg = {"type": "resnet", "channels": 2, "blocks": 1}
    net, opt, step, set_lr, ema = make_trainer(cfg, SHAPE, ema_decay=0.9)
    step(*_batch())
    bn = net.state_dict()["stem.1.running_mean"]
    assert torch.equal(ema["stem.1.running_mean"], bn)


def test_cosine_lr_schedule_values():
    base = 1e-3
    assert cosine_lr(base, 0.0) == 0.0                    # warmup start
    assert abs(cosine_lr(base, 0.015) - 0.0005) < 1e-9    # half of warmup (0.03)
    assert abs(cosine_lr(base, 1.0) - base * 0.02) < 1e-12  # min floor
    assert abs(cosine_lr(base, 1.0, min_lr_frac=0.1) - base * 0.1) < 1e-12
    mids = [cosine_lr(base, f) for f in np.linspace(0.05, 1.0, 50)]
    assert all(a >= b for a, b in zip(mids, mids[1:]))    # monotone decay after warmup
    assert abs(cosine_lr(base, 0.5) - base * 0.51) < 1e-9  # midpoint ~half


def test_setup_torch_threads_noop_and_set():
    assert setup_torch_threads(0) == torch.get_num_threads()
    assert setup_torch_threads(1) == 1
    assert setup_torch_threads(2) == 2
    setup_torch_threads(0)
