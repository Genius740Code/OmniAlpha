"""Candidate assembly for the ordered endgame tournament.

EMA is exported every checkpoint but was never a finalist. eval.include_ema
adds it without changing the champion-last tie-break, and the helper keeps the
file-existence rules in one testable place.
"""
from gai.orchestrator import _endgame_candidates


def _touch(p):
    p.write_text("x")
    return p


def test_candidate_order_and_ema_slot(tmp_path):
    for mins in (5, 10, 15, 20):
        _touch(tmp_path / f"checkpoint_{mins}m.pt")
    _touch(tmp_path / "latest.pt")
    _touch(tmp_path / "latest_ema.pt")
    _touch(tmp_path / "champion.pt")
    _touch(tmp_path / "champion.ts")

    cands = _endgame_candidates(str(tmp_path), n_last=3, include_ema=True)
    names = [c[0] for c in cands]
    assert names == ["checkpoint_15m", "checkpoint_20m", "latest", "ema", "champion"]
    assert cands[-2][1].endswith("latest_ema.ts")
    assert cands[-1][1].endswith("champion.ts")


def test_ema_is_opt_in_and_requires_weights(tmp_path):
    _touch(tmp_path / "latest.pt")
    _touch(tmp_path / "latest_ema.pt")
    assert [c[0] for c in _endgame_candidates(str(tmp_path), 5, include_ema=False)] == ["latest"]
    (tmp_path / "latest_ema.pt").unlink()
    assert [c[0] for c in _endgame_candidates(str(tmp_path), 5, include_ema=True)] == ["latest"]


def test_champion_needs_torchscript_and_n_last_is_clamped(tmp_path):
    for mins in (5, 10, 15):
        _touch(tmp_path / f"checkpoint_{mins}m.pt")
    _touch(tmp_path / "champion.pt")  # no champion.ts -> not a candidate
    cands = _endgame_candidates(str(tmp_path), n_last=1)  # clamped to 2 -> 1 checkpoint
    assert [c[0] for c in cands] == ["checkpoint_15m"]
