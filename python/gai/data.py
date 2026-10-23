"""Reader for engine/selfplay/selfplay.hpp sample files (v2, v3, v4)."""
import numpy as np, struct

def read_samples(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GAIS", "bad magic"
        ver, planes, h, w, actions = struct.unpack("<5I", f.read(20))
        assert ver in (2, 3, 4), f"unsupported version {ver}"
        raw = np.fromfile(f, dtype="<f4")
    in_n = planes * h * w
    rec = in_n + actions + (3 if ver == 4 else (2 if ver == 3 else 1))
    assert raw.size % rec == 0, "truncated file"
    raw = raw.reshape(-1, rec)
    x = raw[:, :in_n].reshape(-1, planes, h, w)
    pi = raw[:, in_n:in_n + actions]
    z = raw[:, in_n + actions]
    full = raw[:, in_n + actions + 1] if ver >= 3 else np.ones(len(z), np.float32)
    # v4 stores the raw search value q (root_value at move time, player-to-move
    # view) after the blended `value` field. Exposed so the (1-lambda)*z+lambda*q
    # mix can be audited on real files without regenerating data; None for v2/v3.
    q = raw[:, in_n + actions + 2] if ver == 4 else None
    return x, pi, z, dict(planes=planes, h=h, w=w, actions=actions, full=full, q=q)

def mirror_augment_connect4(x, pi):
    """Horizontal mirror for Connect-4-like games (policy index == column).
    Returns (x_flip, pi_flip). Generic games: export permutations from C++ instead."""
    xf = x[:, :, :, ::-1].copy()
    pif = pi[:, ::-1].copy()
    return xf, pif

# Registered mirror-symmetry handlers, keyed by the --game name the orchestrator
# passes to the C++ tools. A game may ONLY be listed here when mirroring the
# encoded input and reversing the policy vector is an exact symmetry of the
# game (policy index == mirrored column). Add handlers for other games only
# together with a C++ symmetry export; do not guess.
MIRROR_AUGMENTERS = {"connect4": mirror_augment_connect4}

def maybe_mirror_batch(x, pi, z, full, prob=0.5, rng=np.random, game=None):
    """Randomly mirror ~prob of positions. Returns doubled-or-mixed batch arrays.

    `game` MUST name a registered symmetry handler (see MIRROR_AUGMENTERS).
    Without it this raises instead of mirroring: for a non-symmetric layout the
    mirrored input and reversed policy silently train on corrupted targets,
    which is far worse than a crash at startup.
    """
    if game is None:
        raise ValueError(
            "mirror augmentation requires the game name: pass game=... to "
            "maybe_mirror_batch. Without the right symmetry handler the "
            "mirrored policy targets are silently corrupted.")
    aug = MIRROR_AUGMENTERS.get(game)
    if aug is None:
        raise ValueError(
            f"game '{game}' has no registered mirror-symmetry augmenter "
            f"(known: {sorted(MIRROR_AUGMENTERS)}). Mirroring would corrupt "
            f"its policy targets; register a handler in gai.data.MIRROR_AUGMENTERS "
            f"or set selfplay.augment: 0 in the config.")
    m = rng.rand(len(z)) < prob
    if not m.any():
        return x, pi, z, full
    xf, pif = aug(x[m], pi[m])
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
    def __init__(self, capacity, planes, h, w, actions, track_dups=False):
        self.n = 0; self.cap = capacity; self.pos = 0
        self.x = np.zeros((capacity, planes, h, w), np.float32); self.pi = np.zeros((capacity, actions), np.float32); self.z = np.zeros(capacity, np.float32)
        self.full = np.ones(capacity, np.float32)
        # Duplicate tracking (training.dup_weight>0): content hash per slot, number of
        # ACTIVE slots holding that content (self._dup), and the per-slot dup count
        # (self._cnt) used to downweight repeated samples at draw time. Disabled by
        # default so the plain add()/sample() path stays bit-identical and free.
        self._track = bool(track_dups)
        self._hash = np.zeros(capacity, np.int64)
        self._cnt = np.ones(capacity, np.int64)
        self._dup = {}
        # Cached per-slot acceptance probs for rejection sampling (sample-time
        # dup_weight); invalidated by add() because counts/min-count may change.
        self._acc = None
        self._acc_w = None

    @staticmethod
    def _row_hashes(x, pi, z):
        """Vectorised FNV-1a-style content hash per row -> int64[n]. Never zero (0 is
        reserved for 'empty slot' bookkeeping)."""
        xa = np.ascontiguousarray(x).reshape(len(z), -1).view(np.uint32).astype(np.uint64)
        pa = np.ascontiguousarray(pi).view(np.uint32).astype(np.uint64)
        za = np.ascontiguousarray(z).reshape(len(z), 1).view(np.uint32).astype(np.uint64)
        cols = np.concatenate([xa, pa, za], axis=1)
        h = np.full(len(cols), 0xcbf29ce484222325, np.uint64)
        for c in cols.T:
            h = (h ^ c) * np.uint64(0x100000001b3)
        h = h.astype(np.int64)
        h[h == 0] = np.int64(1)
        return h

    def add(self, x, pi, z, full=None):
        """Ring-buffer insert of a whole batch (vectorised slice assignment).

        Semantics identical to the old per-sample Python loop: samples are written at
        self.pos cyclically, count capped at capacity, pos advances by len(z) mod
        capacity. Uses at most two contiguous slice assignments per array (no
        np.concatenate), so wraparound costs two memcpys instead of a Python loop.
        When len(z) > capacity only the final `cap` samples survive (the ring rewrote
        the earlier ones in place) but pos still advances by the full len(z).
        """
        z = np.asarray(z)
        n = len(z)
        if n == 0:
            return
        if full is None:
            full = np.ones(n, np.float32)
        x = np.asarray(x)[:n]; pi = np.asarray(pi)[:n]; full = np.asarray(full)[:n]
        advance = n
        if n > self.cap:  # only the last `cap` samples survive; start where the loop would
            x, pi, z, full = x[n - self.cap:], pi[n - self.cap:], z[n - self.cap:], full[n - self.cap:]
            n = self.cap
            start = (self.pos + advance - n) % self.cap
        else:
            start = self.pos
        end = start + n
        if self._track:
            h_new = self._row_hashes(np.ascontiguousarray(x), np.ascontiguousarray(pi),
                                     np.ascontiguousarray(z))
            self._acc_w = None  # counts may change -> cached acceptance probs stale
            if end <= self.cap:
                ov = range(start, end)
            else:
                ov = list(range(start, self.cap)) + list(range(0, n - (self.cap - start)))
            for s in ov:  # slots being overwritten: retire their old content
                oh = int(self._hash[s])
                if oh:
                    self._dup[oh] = self._dup.get(oh, 1) - 1
                    if self._dup[oh] <= 0:
                        del self._dup[oh]
            for s, h in zip(ov, h_new):  # register the incoming content
                hi = int(h)
                self._hash[s] = hi
                self._dup[hi] = self._dup.get(hi, 0) + 1
            for s, h in zip(ov, h_new):  # per-slot count = FINAL active copies
                self._cnt[s] = self._dup[int(h)]
        if end <= self.cap:  # contiguous: one memcpy slice per array
            self.x[start:end] = x; self.pi[start:end] = pi
            self.z[start:end] = z; self.full[start:end] = full
        else:  # wraparound: two memcpy slices
            first = self.cap - start
            self.x[start:] = x[:first]; self.pi[start:] = pi[:first]
            self.z[start:] = z[:first]; self.full[start:] = full[:first]
            rem = n - first
            self.x[:rem] = x[first:]; self.pi[:rem] = pi[first:]
            self.z[:rem] = z[first:]; self.full[:rem] = full[first:]
        self.pos = (self.pos + advance) % self.cap
        self.n = min(self.n + advance, self.cap)
    def sample(self, batch, rng=np.random, recency=0.0, dup_weight=0.0):
        """Draw `batch` indices and return (x, pi, z, full).

        recency in [0,1] is a mixture: that fraction of the batch is drawn from the newest
        half-window of the ring (anti-forgetting of the current policy) and the remainder
        from the older half. So recency=0.7 means ~70% recent / ~30% older, and recency=0
        reproduces the original uniform-over-all behaviour.

        dup_weight>0 (requires track_dups=True at construction) downweights duplicate
        content: draw probability ∝ count**(-dup_weight), so dup_weight=1 gives each
        distinct position equal total mass regardless of how many copies exist, and
        dup_weight=0 is the untouched uniform path.
        """
        if self.n <= 0:
            raise ValueError("ReplayBuffer.sample on empty buffer")
        dup_weight = float(dup_weight or 0.0)
        if dup_weight > 0.0 and self._track:
            # Rejection sampling toward p(i) ∝ count(i)**(-dup_weight): draw uniform,
            # accept with prob (c_min/c_i)**dup_weight, refill rejects. Exact target
            # distribution, O(batch) per round; acceptance vector cached across steps
            # (invalidated by add()).
            if self._acc_w != dup_weight or self._acc is None or len(self._acc) != self.n:
                c = self._cnt[:self.n].astype(np.float64)
                self._acc = (c.min() / c) ** dup_weight
                self._acc_w = dup_weight
            acc = self._acc
            idx = np.empty(batch, dtype=np.int64)
            need = np.arange(batch)
            for _ in range(16):  # ~1 round when few duplicates; 16 rounds is overkill
                draw = rng.randint(0, self.n, size=len(need))
                keep = rng.rand(len(need)) <= acc[draw]
                idx[need[keep]] = draw[keep]
                need = need[~keep]
                if len(need) == 0:
                    break
            if len(need):  # fallback: accept as drawn (distribution slightly off, rare)
                idx[need] = rng.randint(0, self.n, size=len(need))
            return self.x[idx], self.pi[idx], self.z[idx], self.full[idx]
        recency = float(recency)
        if recency <= 0.0:
            idx = rng.randint(0, self.n, size=batch)
            return self.x[idx], self.pi[idx], self.z[idx], self.full[idx]
        if recency > 1.0:
            recency = 1.0
        half = max(1, self.n // 2)
        k_new = int(round(recency * batch))
        idx = np.empty(batch, dtype=np.int64)
        if k_new > 0:
            # Newest `half` entries: ring slice ending at self.pos-1.
            start = (self.pos - half) % self.cap
            idx[:k_new] = (start + rng.randint(0, half, size=k_new)) % self.cap
        if k_new < batch:
            old_n = self.n - half
            fill = batch - k_new
            if old_n > 0:
                old_start = (self.pos - self.n) % self.cap
                idx[k_new:] = (old_start + rng.randint(0, old_n, size=fill)) % self.cap
            else:  # buffer holds a single entry
                idx[k_new:] = (self.pos - 1) % self.cap
        return self.x[idx], self.pi[idx], self.z[idx], self.full[idx]
