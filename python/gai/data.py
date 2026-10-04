"""Reader for engine/selfplay/selfplay.hpp sample files (v2 and v3)."""
import numpy as np, struct

def read_samples(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GAIS", "bad magic"
        ver, planes, h, w, actions = struct.unpack("<5I", f.read(20))
        assert ver in (2, 3), f"unsupported version {ver}"
        raw = np.fromfile(f, dtype="<f4")
    in_n = planes * h * w
    rec = in_n + actions + (2 if ver == 3 else 1)
    assert raw.size % rec == 0, "truncated file"
    raw = raw.reshape(-1, rec)
    x = raw[:, :in_n].reshape(-1, planes, h, w)
    pi = raw[:, in_n:in_n + actions]
    z = raw[:, in_n + actions]
    full = raw[:, in_n + actions + 1] if ver == 3 else np.ones(len(z), np.float32)
    return x, pi, z, dict(planes=planes, h=h, w=w, actions=actions, full=full)

def mirror_augment_connect4(x, pi):
    """Horizontal mirror for Connect-4-like games (policy index == column).
    Returns (x_flip, pi_flip). Generic games: export permutations from C++ instead."""
    xf = x[:, :, :, ::-1].copy()
    pif = pi[:, ::-1].copy()
    return xf, pif

def maybe_mirror_batch(x, pi, z, full, prob=0.5, rng=np.random):
    """Randomly mirror ~prob of positions (Connect-4). Returns doubled-or-mixed batch arrays."""
    m = rng.rand(len(z)) < prob
    if not m.any():
        return x, pi, z, full
    xf, pif = mirror_augment_connect4(x[m], pi[m])
    return (np.concatenate([x, xf]), np.concatenate([pi, pif]),
            np.concatenate([z, z[m]]), np.concatenate([full, full[m]]))

def split_held_out(x, pi, z, full, holdout_frac=0.1):
    """Split off the last holdout_frac of the batch as held-out data.
    The held-out portion is NEVER added to ReplayBuffer.
    Returns (train_x, train_pi, train_z, train_full, heldout_x, heldout_pi, heldout_z, heldout_full).
    """
    n = len(z)
    holdout_size = int(n * holdout_frac)
    if holdout_size <= 0 or holdout_size >= n:
        return x, pi, z, full, np.empty((0, *x.shape[1:])), np.empty((0, *pi.shape[1:])), np.empty((0,)), np.ones((0,), np.float32)
    train_end = n - holdout_size
    return (x[:train_end], pi[:train_end], z[:train_end], full[:train_end],
            x[train_end:], pi[train_end:], z[train_end:], full[train_end:])

class ReplayBuffer:
    """Fixed-capacity in-memory ring buffer (competition mode keeps everything in RAM)."""
    def __init__(self, capacity, planes, h, w, actions):
        self.n = 0; self.cap = capacity; self.pos = 0
        self.x = np.zeros((capacity, planes, h, w), np.float32); self.pi = np.zeros((capacity, actions), np.float32); self.z = np.zeros(capacity, np.float32)
        self.full = np.ones(capacity, np.float32)
    def add(self, x, pi, z, full=None):
        if full is None: full = np.ones(len(z), np.float32)
        for i in range(len(z)):
            self.x[self.pos], self.pi[self.pos], self.z[self.pos], self.full[self.pos] = x[i], pi[i], z[i], full[i]
            self.pos = (self.pos + 1) % self.cap; self.n = min(self.n + 1, self.cap)
    def sample(self, batch, rng=np.random):
        idx = rng.randint(0, self.n, size=batch); return self.x[idx], self.pi[idx], self.z[idx], self.full[idx]
