"""Replay-buffer persistence for long-run resume (`training.resume_replay`).

A 10-hour run that loses its replay window after a crash spends the next
iterations re-filling it with weak/duplicate data. save() writes only the live
ring entries oldest-first; load() re-adds through add() so duplicate bookkeeping
and recency order are rebuilt instead of persisted twice.
"""
import numpy as np
import pytest

from gai.data import ReplayBuffer


def _filled(cap, n, planes=1, h=1, w=1, actions=3, start=0):
    b = ReplayBuffer(cap, planes, h, w, actions)
    xs = np.zeros((n, planes, h, w), np.float32)
    for i in range(n):
        xs[i].flat[0] = float(start + i)
    pi = np.zeros((n, actions), np.float32)
    pi[:, 0] = 1.0
    z = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    full = np.ones(n, np.float32)
    full[::4] = 0.0
    b.add(xs, pi, z, full)
    return b


def _active_tags(b):
    return b.x[b._chronological_indices(), 0, 0, 0]


def test_roundtrip_preserves_wrapped_order_and_flags(tmp_path):
    b = _filled(5, 3, start=0)          # tags 0,1,2
    b.add(*_batch_tags(4, start=3))     # wrap: only tags 2..6 remain
    assert b.n == 5
    p = str(tmp_path / "replay.npz")
    assert b.save(p) == 5

    b2 = ReplayBuffer(5, 1, 1, 1, 3)
    assert b2.load(p) == 5
    assert b2.n == 5
    assert _active_tags(b2).tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert np.array_equal(b2.full[:b2.n], b.full[b._chronological_indices()])
    assert np.array_equal(b2.z[:b2.n], b.z[b._chronological_indices()])
    # recency order is rebuilt: newest half must be tags 5,6 only.
    x, _, _, _ = b2.sample(200, rng=np.random.RandomState(0), recency=1.0)
    assert x[:, 0, 0, 0].min() >= 5.0


def _batch_tags(n, start=0, actions=3):
    xs = np.zeros((n, 1, 1, 1), np.float32)
    for i in range(n):
        xs[i].flat[0] = float(start + i)
    pi = np.zeros((n, actions), np.float32)
    pi[:, 0] = 1.0
    z = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    full = np.ones(n, np.float32)
    return xs, pi, z, full


def test_duplicate_tracking_is_rebuilt_exactly(tmp_path):
    b = ReplayBuffer(8, 1, 1, 1, 3, track_dups=True)
    xs = np.zeros((6, 1, 1, 1), np.float32)
    xs[:4].flat[0] = 1.0
    xs[4:].flat[0] = 2.0
    b.add(xs, np.zeros((6, 3), np.float32), np.zeros(6, np.float32))
    p = str(tmp_path / "dup.npz")
    b.save(p)

    b2 = ReplayBuffer(8, 1, 1, 1, 3, track_dups=True)
    b2.load(p)
    assert b2._dup == b._dup
    assert np.array_equal(b2._cnt[:b2.n], b._cnt[:b.n])
    assert np.array_equal(b2._hash[:b2.n], b._hash[:b.n])


def test_load_into_smaller_capacity_keeps_newest(tmp_path):
    b = _filled(6, 5, start=1)          # tags 1..5
    p = str(tmp_path / "r.npz")
    b.save(p)
    b2 = ReplayBuffer(3, 1, 1, 1, 3)
    assert b2.load(p) == 5
    assert b2.n == 3
    assert sorted(b2.x[:b2.n, 0, 0, 0].tolist()) == [3.0, 4.0, 5.0]


def test_empty_buffer_roundtrip(tmp_path):
    b = ReplayBuffer(4, 1, 1, 1, 3)
    p = str(tmp_path / "empty.npz")
    assert b.save(p) == 0
    b2 = ReplayBuffer(4, 1, 1, 1, 3)
    assert b2.load(p) == 0
    assert b2.n == 0


def test_shape_mismatch_is_loud(tmp_path):
    b = _filled(4, 2, planes=1, h=2, w=2)
    p = str(tmp_path / "shape.npz")
    b.save(p)
    b2 = ReplayBuffer(4, 1, 1, 1, 3)
    with pytest.raises(ValueError, match="shape"):
        b2.load(p)
