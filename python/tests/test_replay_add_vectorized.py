"""ReplayBuffer.add vectorisation vs the reference per-sample loop (python/gai/data.py).

The vectorised ring-buffer insert must be bit-identical to the old sample-by-sample
Python loop — same contents, same final pos, same capped count — including the exact
wraparound case (batch crosses the capacity boundary) and the overflow case (batch
larger than the whole buffer).
"""
import numpy as np
import pytest
from gai.data import ReplayBuffer


class _LoopReplayBuffer(ReplayBuffer):
    """Reference implementation: the original per-sample Python loop."""

    def add(self, x, pi, z, full=None):
        if full is None:
            full = np.ones(len(z), np.float32)
        for i in range(len(z)):
            self.x[self.pos], self.pi[self.pos], self.z[self.pos], self.full[self.pos] = \
                x[i], pi[i], z[i], full[i]
            self.pos = (self.pos + 1) % self.cap
            self.n = min(self.n + 1, self.cap)


def _random_batch(rng, n, planes, h, w, actions):
    return (rng.randn(n, planes, h, w).astype(np.float32),
            rng.randn(n, actions).astype(np.float32),
            rng.randn(n).astype(np.float32),
            (rng.rand(n) > 0.3).astype(np.float32))


def _assert_equal(a, b):
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.pi, b.pi)
    np.testing.assert_array_equal(a.z, b.z)
    np.testing.assert_array_equal(a.full, b.full)
    assert a.pos == b.pos, (a.pos, b.pos)
    assert a.n == b.n, (a.n, b.n)


def test_matches_reference_loop_on_random_sequences():
    rng = np.random.RandomState(42)
    for _ in range(8):
        cap = int(rng.randint(1, 40))
        planes, h, w, actions = 2, 3, 3, 7
        vec = ReplayBuffer(cap, planes, h, w, actions)
        ref = _LoopReplayBuffer(cap, planes, h, w, actions)
        for _ in range(int(rng.randint(1, 6))):
            # sizes cover 0, partial fill, exact wraparound and overflow (n > cap)
            n = int(rng.randint(0, cap * 2 + 3))
            batch = _random_batch(rng, n, planes, h, w, actions)
            vec.add(*batch)
            ref.add(*batch)
            _assert_equal(vec, ref)


def test_exact_wraparound_batch():
    cap, planes, h, w, actions = 10, 1, 2, 2, 4
    vec = ReplayBuffer(cap, planes, h, w, actions)
    ref = _LoopReplayBuffer(cap, planes, h, w, actions)
    rng = np.random.RandomState(7)
    # first add leaves pos=7; second add (6 samples) crosses the boundary at 10
    for batch in (_random_batch(rng, 7, planes, h, w, actions),
                  _random_batch(rng, 6, planes, h, w, actions)):
        vec.add(*batch)
        ref.add(*batch)
        _assert_equal(vec, ref)
    assert vec.pos == (7 + 6) % cap == 3


def test_overflow_batch_keeps_last_cap_samples():
    cap, planes, h, w, actions = 8, 1, 1, 1, 3
    vec = ReplayBuffer(cap, planes, h, w, actions)
    ref = _LoopReplayBuffer(cap, planes, h, w, actions)
    rng = np.random.RandomState(11)
    batch = _random_batch(rng, 25, planes, h, w, actions)
    vec.add(*batch)
    ref.add(*batch)
    _assert_equal(vec, ref)
    assert vec.n == cap
    # Ring layout derived independently from the loop's index math: sample
    # (advance - cap + j) lands at position (start + j) % cap with
    # start = (pos0 + advance - cap) % cap — i.e. the last `cap` samples survive.
    advance, pos0 = 25, 0
    start = (pos0 + advance - cap) % cap
    expected = np.empty(cap, np.float32)
    for j in range(cap):
        expected[(start + j) % cap] = batch[2][advance - cap + j]
    np.testing.assert_array_equal(vec.z, expected)


def test_exact_capacity_fill_from_zero():
    cap, planes, h, w, actions = 16, 2, 2, 2, 5
    vec = ReplayBuffer(cap, planes, h, w, actions)
    ref = _LoopReplayBuffer(cap, planes, h, w, actions)
    rng = np.random.RandomState(13)
    batch = _random_batch(rng, cap, planes, h, w, actions)
    vec.add(*batch)
    ref.add(*batch)
    _assert_equal(vec, ref)
    assert vec.pos == 0 and vec.n == cap


def test_zero_length_add_is_noop():
    b = ReplayBuffer(4, 1, 1, 1, 2)
    b.add(np.zeros((0, 1, 1, 1), np.float32), np.zeros((0, 2), np.float32), np.zeros((0,), np.float32))
    assert b.n == 0 and b.pos == 0


def test_full_defaults_to_ones():
    b = ReplayBuffer(4, 1, 1, 1, 2)
    b.add(np.ones((3, 1, 1, 1), np.float32), np.zeros((3, 2), np.float32), np.ones(3, np.float32))
    np.testing.assert_array_equal(b.full, np.ones(4, np.float32))


def test_float64_inputs_cast_into_float32_buffer():
    b = ReplayBuffer(4, 1, 1, 1, 2)
    b.add(np.ones((3, 1, 1, 1)), np.zeros((3, 2)), np.ones(3))  # float64 in
    assert b.x.dtype == np.float32 and b.z.dtype == np.float32
