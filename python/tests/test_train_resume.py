"""Long-run resume plumbing: atomic training state + checkpoint retention.

A 10-hour run must survive a crash without restarting the net, optimizer moments,
EMA average, iteration counter or wall-clock LR phase from zero. Replay data is
deliberately not part of the state file (it refills in one or two iterations),
but optimizer state is what avoids a loss spike at the restart boundary.
"""
import numpy as np
import torch

from gai.train import make_trainer, save_train_state, load_train_state

SHAPE = dict(planes=1, h=2, w=2, actions=3)
CFG = {"type": "mlp", "hidden": 8, "layers": 1}


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 1, 2, 2)).astype(np.float32)
    pi = rng.random((n, 3)).astype(np.float32)
    pi /= pi.sum(-1, keepdims=True)
    z = rng.random(n).astype(np.float32) * 2 - 1
    return x, pi, z


def test_train_state_roundtrip_restores_weights_and_optimizer(tmp_path):
    net, opt, step, set_lr, ema = make_trainer(
        CFG, SHAPE, ema_decay=0.8, device="cpu")
    step(*_batch())
    path = str(tmp_path / "train_state.pt")
    save_train_state(path, net, opt, ema, it=7, elapsed_s=123.5,
                     extra={"shape": SHAPE})

    net2, opt2, step2, set_lr2, ema2 = make_trainer(
        CFG, SHAPE, ema_decay=0.8, device="cpu")
    meta = load_train_state(path, net=net2, opt=opt2, device="cpu")
    assert meta["iter"] == 7
    assert abs(meta["elapsed_s"] - 123.5) < 1e-6
    assert meta["shape"] == SHAPE
    assert meta["format"] == 1
    for k, v in net.state_dict().items():
        assert torch.allclose(net2.state_dict()[k], v), k
    # Optimizer moments must come back too, otherwise Adam restarts cold.
    sd1 = opt.state_dict()
    sd2 = opt2.state_dict()
    assert set(sd1["state"]) == set(sd2["state"])
    for k in sd1["state"]:
        for name, t in sd1["state"][k].items():
            assert torch.allclose(sd2["state"][k][name], t), (k, name)


def test_ema_init_restores_average_into_updater(tmp_path):
    net, opt, step, set_lr, ema = make_trainer(
        CFG, SHAPE, ema_decay=0.7, device="cpu")
    step(*_batch(seed=3))
    path = str(tmp_path / "state.pt")
    save_train_state(path, net, opt, ema, it=1, elapsed_s=10.0)
    meta = load_train_state(path, device="cpu", load_weights=False)

    net2, opt2, step2, set_lr2, ema2 = make_trainer(
        CFG, SHAPE, ema_decay=0.7, device="cpu", ema_init=meta["ema"])
    for k, v in ema.items():
        assert torch.allclose(ema2[k], v), k
