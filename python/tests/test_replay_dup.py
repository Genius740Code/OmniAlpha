"""Round C train-side: ReplayBuffer duplicate downweighting (training.dup_weight)."""
import numpy as np
import pytest
from gai.data import ReplayBuffer


def _mk(planes=1, h=1, w=1, actions=3):
    return planes, h, w, actions


def test_dup_counts_same_content():
    p, h, w, a = _mk()
    b = ReplayBuffer(8, p, h, w, a, track_dups=True)
    xs = np.zeros((4, p, h, w), np.float32); xs[:, 0, 0, 0] = 1.0  # identical rows
    b.add(xs, np.zeros((4, a), np.float32), np.zeros(4, np.float32))
    assert (b._cnt[:4] == 4).all()
    assert len(b._dup) == 1


def test_distinct_rows_count_one():
    p, h, w, a = _mk()
    b = ReplayBuffer(8, p, h, w, a, track_dups=True)
    xs = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
    b.add(xs, np.zeros((4, a), np.float32), np.zeros(4, np.float32))
    assert (b._cnt[:4] == 1).all()
    assert len(b._dup) == 4


def test_ring_overwrite_retires_old_content():
    """Identical rows: retire-on-overwrite keeps dict and per-slot counts exact."""
    p, h, w, a = _mk()
    b = ReplayBuffer(4, p, h, w, a, track_dups=True)
    xs = np.full((4, 1, 1, 1), 1.0, np.float32)  # 4 IDENTICAL rows
    b.add(xs, np.zeros((4, a), np.float32), np.zeros(4, np.float32))
    assert (b._cnt == 4).all()
    b.add(xs, np.zeros((4, a), np.float32), np.zeros(4, np.float32))  # full overwrite
    assert (b._cnt == 4).all()  # still 4 active copies of the same content
    assert sum(b._dup.values()) == 4  # invariant: dict counts active slots
    # third add replaces all 4 slots with 4 NEW identical rows (content 9.0):
    # old hash retired to zero, new hash active with 4 copies
    b.add(np.full((4, 1, 1, 1), 9.0, np.float32), np.zeros((4, a), np.float32), np.zeros(4, np.float32))
    assert (b._cnt == 4).all()
    assert sum(b._dup.values()) == 4
    assert len(b._dup) == 1


def test_wraparound_counts_stay_consistent():
    p, h, w, a = _mk()
    b = ReplayBuffer(4, p, h, w, a, track_dups=True)
    rng = np.random.RandomState(0)
    for it in range(6):
        n = 3
        n0 = b.n
        b.add(rng.rand(n, p, h, w).astype(np.float32), np.zeros((n, a), np.float32),
              np.zeros(n, np.float32))
        assert b.n == min(b.cap, n0 + n)
        assert sum(b._dup.values()) == b.n
        assert (b._cnt[:b.n] >= 1).all()
        # each slot's stored hash must match its content
        for s in range(b.n):
            assert b._hash[s] == int(ReplayBuffer._row_hashes(b.x[s:s+1], b.pi[s:s+1], b.z[s:s+1])[0])


def test_weighted_sampling_downweights_duplicates():
    p, h, w, a = _mk()
    b = ReplayBuffer(8, p, h, w, a, track_dups=True)
    xs = np.zeros((6, p, h, w), np.float32); xs[:, 0, 0, 0] = 1.0  # 6 copies of one row
    b.add(xs, np.zeros((6, a), np.float32), np.zeros(6, np.float32))
    # one distinct row
    b.add(np.full((1, 1, 1, 1), 5.0, np.float32), np.zeros((1, a), np.float32), np.zeros(1, np.float32))
    assert b.n == 7
    idx = b.sample(20000, rng=np.random.RandomState(0), dup_weight=1.0)[0]
    hits_unique = (idx[:, 0, 0, 0] == 5.0).mean()
    # dup_weight=1 gives each distinct CONTENT equal mass: 2 distinct contents ->
    # the lone unique row should get ~1/2 of draws despite holding 1/7 of slots
    # (uniform would give ~1/7 ~= 0.14).
    assert 0.4 < hits_unique < 0.6


def test_flag_off_path_untouched():
    p, h, w, a = _mk()
    plain = ReplayBuffer(8, p, h, w, a)
    assert not plain._track
    xs = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
    plain.add(xs, np.zeros((4, a), np.float32), np.zeros(4, np.float32))
    x1, pi1, z1, f1 = plain.sample(4, rng=np.random.RandomState(1))
    # same seed on a tracked buffer with dup_weight=0 must give identical draws
    tr = ReplayBuffer(8, p, h, w, a, track_dups=True)
    tr.add(xs, np.zeros((4, a), np.float32), np.zeros(4, np.float32))
    x2, pi2, z2, f2 = tr.sample(4, rng=np.random.RandomState(1), dup_weight=0.0)
    assert np.array_equal(x1, x2) and np.array_equal(z1, z2)


def test_tracked_and_plain_buffers_store_identical_data():
    p, h, w, a = _mk(2, 2, 2)
    rng = np.random.RandomState(3)
    xs = rng.rand(5, 2, 2, 2).astype(np.float32)
    pis = rng.rand(5, 2).astype(np.float32)
    zs = rng.rand(5).astype(np.float32)
    plain = ReplayBuffer(8, 2, 2, 2, 2)
    tr = ReplayBuffer(8, 2, 2, 2, 2, track_dups=True)
    for b in (plain, tr):
        b.add(xs, pis, zs)
        b.add(xs, pis, zs)
    assert np.array_equal(plain.x, tr.x) and np.array_equal(plain.pi, tr.pi)
    assert np.array_equal(plain.z, tr.z)
    x1, _, z1, _ = plain.sample(16, rng=np.random.RandomState(5))
    x2, _, z2, _ = tr.sample(16, rng=np.random.RandomState(5))
    assert np.array_equal(x1, x2) and np.array_equal(z1, z2)


def test_hash_never_zero():
    p, h, w, a = _mk()
    hsh = ReplayBuffer._row_hashes(np.zeros((3, p, h, w), np.float32),
                                   np.zeros((3, a), np.float32), np.zeros(3, np.float32))
    assert (hsh != 0).all()
