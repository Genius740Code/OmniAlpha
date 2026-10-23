"""Mirror-augmentation generality: unknown games must fail loudly, not corrupt.

maybe_mirror_batch used to mirror ANY game's inputs and reverse its policy
vector unconditionally. For a game whose encoded input is not left-right
symmetric (or whose policy index is not the mirrored column), that silently
writes corrupted policy targets into the replay buffer. It must raise
ValueError unless the game has a registered symmetry handler.
"""
import numpy as np
import pytest
from gai.data import maybe_mirror_batch, mirror_augment_connect4, MIRROR_AUGMENTERS


class _Rng:
    """Deterministic rng: rows flagged True draw 0.9 (< prob) so they get
    mirrored, others draw 0.0 (>= prob for any prob < 1)."""

    def __init__(self, keep):
        self.keep = np.asarray(keep, dtype=bool)

    def rand(self, n):
        out = np.zeros(n, dtype=np.float32)
        for i in range(min(n, len(self.keep))):
            out[i] = 0.9 if self.keep[i] else 0.0
        return out


def _batch(n=4):
    x = np.arange(n * 2 * 1 * 3, dtype=np.float32).reshape(n, 2, 1, 3)
    pi = np.full((n, 3), 1 / 3, np.float32)
    z = np.linspace(-1, 1, n).astype(np.float32)
    full = np.ones(n, np.float32)
    return x, pi, z, full


def test_connect4_mirror_unchanged_behaviour():
    x, pi, z, full = _batch(4)
    xa, pa, za, fa = maybe_mirror_batch(x, pi, z, full, prob=1.0,
                                       rng=_Rng([1, 1, 1, 1]), game="connect4")
    # every position mirrored -> doubled batch, mirrored rows appended
    assert len(za) == 8
    xf, pif = mirror_augment_connect4(x, pi)
    assert np.array_equal(xa[4:], xf) and np.array_equal(pa[4:], pif)
    assert np.array_equal(za[4:], z) and np.array_equal(fa[4:], full)


def test_connect4_mirror_none_selected_is_noop():
    x, pi, z, full = _batch(4)
    # every row draws 0.9 which is >= prob=0.5: nothing mirrored -> untouched
    xa, pa, za, fa = maybe_mirror_batch(x, pi, z, full, prob=0.5,
                                        rng=_Rng([1, 1, 1, 1]), game="connect4")
    assert xa is x and za is z  # nothing mirrored: arrays returned untouched


def test_missing_game_argument_raises():
    x, pi, z, full = _batch(4)
    with pytest.raises(ValueError, match="game"):
        maybe_mirror_batch(x, pi, z, full, rng=_Rng([1, 1, 1, 1]))


def test_unknown_game_raises_not_silently_mirrors():
    x, pi, z, full = _batch(4)
    with pytest.raises(ValueError, match="breakthrough"):
        maybe_mirror_batch(x, pi, z, full, rng=_Rng([1, 1, 1, 1]), game="breakthrough")
    # and nothing was returned / mirrored: the call must abort, not fall back
    with pytest.raises(ValueError):
        maybe_mirror_batch(x, pi, z, full, rng=_Rng([1, 1, 1, 1]), game="template")


def test_registry_only_lists_verified_symmetric_games():
    # connect4 is the only game verified to have exact horizontal-mirror
    # symmetry (policy index == column). Template declares kNumSymmetries=1,
    # so it must NOT be in the registry.
    assert "connect4" in MIRROR_AUGMENTERS
    assert "template" not in MIRROR_AUGMENTERS


def test_error_message_mentions_the_fix():
    x, pi, z, full = _batch(2)
    with pytest.raises(ValueError) as ei:
        maybe_mirror_batch(x, pi, z, full, rng=_Rng([1, 1]), game="chess960")
    msg = str(ei.value)
    assert "MIRROR_AUGMENTERS" in msg and "augment" in msg
