"""Sample-file format + the q-sign audit, end to end.

read_samples must parse the v4 binary layout field-by-field (input, policy,
value, full, q) with no misalignment, and the q sign must be correct from the
NEXT-MOVER'S view through the whole self-play emission path. The lambda sweep
proves it live: value_lambda only enters the post-hoc mix
    value = (1-lambda)*outcome + lambda*q
so with identical seeds the three runs produce bit-identical trajectories and
policies, and only the stored value differs. Then:
  * z@l=0.5 == 0.5*z@l=0 + 0.5*q   (mix formula, bit-for-bit)
  * q must correlate POSITIVELY with the eventual outcome from the mover's
    own view (the v7 bug anti-correlated it).
"""
import os
import struct
import subprocess

import numpy as np
import pytest

from gai.data import read_samples

BUILD_SELfPLAY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "build", "selfplay")

needs_build = pytest.mark.skipif(not os.path.exists(BUILD_SELfPLAY),
                                reason="build/selfplay not available")


def _write_file(path, ver, planes, h, w, actions, records):
    """records: list of (input flat list, policy list, value, full, q_or_None)."""
    with open(path, "wb") as f:
        f.write(b"GAIS")
        f.write(struct.pack("<5I", ver, planes, h, w, actions))
        for inp, pol, value, full, q in records:
            row = list(inp) + list(pol) + [value, full] + ([q] if ver == 4 else [])
            f.write(struct.pack(f"<{len(row)}f", *row))


def test_read_samples_v4_fields_bit_exact(tmp_path):
    p = tmp_path / "v4.bin"
    _write_file(str(p), 4, 1, 1, 2, 3, [
        ([1.0, 2.0], [0.5, 0.3, 0.2], 0.9, 1.0, 0.25),
        ([3.0, 4.0], [0.0, 1.0, 0.0], -0.7, 0.0, -1.0),
    ])
    x, pi, z, meta = read_samples(str(p))
    assert x.shape == (2, 1, 1, 2)
    assert np.array_equal(x.reshape(2, 2), np.array([[1.0, 2.0], [3.0, 4.0]], np.float32))
    assert np.array_equal(pi, np.array([[0.5, 0.3, 0.2], [0, 1, 0]], np.float32))
    assert np.array_equal(z, np.array([0.9, -0.7], np.float32))
    assert np.array_equal(meta["full"], np.array([1.0, 0.0], np.float32))
    # q is exposed separately and must NOT collide with value/full
    assert np.array_equal(meta["q"], np.array([0.25, -1.0], np.float32))


def test_read_samples_v3_has_no_q(tmp_path):
    p = tmp_path / "v3.bin"
    _write_file(str(p), 3, 1, 1, 2, 3, [([1.0, 2.0], [0.5, 0.5, 0.0], -1.0, 1.0, None)])
    x, pi, z, meta = read_samples(str(p))
    assert z[0] == -1.0 and meta["full"][0] == 1.0
    assert meta["q"] is None  # older format: no q stored, reader says so


def _selfplay(tmp_path, lam, games=40, sims=100, seed=42):
    out = str(tmp_path / f"lam_{lam}.bin")
    cmd = [BUILD_SELfPLAY, "--game", "connect4", "--games", str(games),
           "--sims", str(sims), "--threads", "1", "--seed", str(seed),
           "--value-lambda", str(lam), "--out", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return read_samples(out)


@needs_build
def test_q_sign_and_mix_formula_bit_exact(tmp_path):
    x0, pi0, z0, m0 = _selfplay(tmp_path, 0.0)
    xh, pih, zh, mh = _selfplay(tmp_path, 0.5)
    x1, pi1, z1, m1 = _selfplay(tmp_path, 1.0)
    # lambda never touches the RNG: identical trajectories bit-for-bit
    assert len(z0) == len(zh) == len(z1) and len(z0) > 100
    assert np.array_equal(x0, xh) and np.array_equal(x0, x1)
    assert np.array_equal(pi0, pih) and np.array_equal(pi0, pi1)
    assert np.array_equal(m0["q"], mh["q"]) and np.array_equal(m0["q"], m1["q"])
    # mix formula: value(l) = (1-l)*outcome + l*q, verified against stored q
    mix_err = np.abs(zh - (0.5 * z0 + 0.5 * mh["q"])).max()
    assert mix_err < 1e-6
    # q(l=1) must equal the stored q exactly (l=1 makes value == q)
    assert np.abs(z1 - m1["q"]).max() == 0.0
    # SIGN: q from the mover's view must predict the mover's own outcome
    q, outc = m1["q"], z0
    won, lost = outc == 1.0, outc == -1.0
    assert won.sum() > 20 and lost.sum() > 20  # enough decisive games to judge
    assert q[won].mean() > q[lost].mean() + 0.05
    assert float(np.corrcoef(outc, q)[0, 1]) > 0.2
    # shortcut samples (full=0, win-in-1) carry certain q=+1 from the mover's view
    shortcut = (m0["full"] == 0.0) & (np.abs(m0["q"] - 1.0) < 1e-6)
    if shortcut.any():
        assert (z0[shortcut] == 1.0).all()  # the mover really won those games
