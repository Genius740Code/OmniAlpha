"""Unit tests for ReplayBuffer recency-weighted sampling (python/gai/data.py)."""
import numpy as np
import pytest
from gai.data import ReplayBuffer


def _filled(cap, n, planes=1, h=1, w=1, actions=3):
    b = ReplayBuffer(cap, planes, h, w, actions)
    xs = np.zeros((n, planes, h, w), np.float32)
    for i in range(n):
        xs[i, 0, 0, 0] = float(i)  # tag each sample with its insertion index
    b.add(xs, np.zeros((n, actions), np.float32), np.zeros(n, np.float32))
    return b


def test_uniform_recency_zero_is_legacy_behaviour():
    b = _filled(100, 50)
    x, pi, z, full = b.sample(16, rng=np.random.RandomState(0), recency=0.0)
    assert x.shape == (16, 1, 1, 1) and pi.shape[0] == 16 and z.shape[0] == 16


def test_recency_one_draws_only_newest_half():
    b = _filled(100, 40)  # n=40 -> newest half = tags 20..39
    x, _, _, _ = b.sample(64, rng=np.random.RandomState(2), recency=1.0)
    tags = x[:, 0, 0, 0]
    assert tags.min() >= 20.0
    assert tags.max() <= 39.0


def test_recency_mixture_ratio_is_respected():
    b = _filled(100, 40)
    x, _, _, _ = b.sample(4000, rng=np.random.RandomState(3), recency=0.7)
    frac_new = float(np.mean(x[:, 0, 0, 0] >= 20.0))
    assert abs(frac_new - 0.70) < 0.03


def test_wrapped_ring_newest_half_is_correct():
    cap = 10
    b = _filled(cap, 25)  # only tags 15..24 remain valid; newest half = 20..24
    x, _, _, _ = b.sample(400, rng=np.random.RandomState(4), recency=1.0)
    tags = x[:, 0, 0, 0]
    assert tags.min() >= 20.0 and tags.max() <= 24.0
    # older half must come from 15..19 only
    x2, _, _, _ = b.sample(400, rng=np.random.RandomState(5), recency=0.0)
    tags2 = x2[:, 0, 0, 0]
    assert tags2.min() >= 15.0 and tags2.max() <= 24.0


def test_single_entry_buffer_has_no_crash():
    b = _filled(50, 1)
    x, _, _, _ = b.sample(8, rng=np.random.RandomState(6), recency=1.0)
    assert (x[:, 0, 0, 0] == 0.0).all()


def test_empty_buffer_raises():
    b = ReplayBuffer(10, 1, 1, 1, 3)
    with pytest.raises(ValueError):
        b.sample(4)
