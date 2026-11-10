"""Checkpoint retention config for long runs.

checkpoint.keep is the primary knob; champion.keep_last_n_checkpoints is the
legacy long_term.yaml key that used to be silently ignored. Keeping more
checkpoints lets the ordered endgame tournament pick a mid-run peak instead of
only the last few minutes.
"""
from gai.orchestrator import _checkpoint_keep, _prune_checkpoints


def test_checkpoint_keep_config_fallbacks():
    assert _checkpoint_keep({}) == 5
    assert _checkpoint_keep({"checkpoint": {"keep": 20}}) == 20
    assert _checkpoint_keep({"champion": {"keep_last_n_checkpoints": 12}}) == 12
    assert _checkpoint_keep({"checkpoint": {"keep": 0}}) == 0
    assert _checkpoint_keep({"checkpoint": {"keep": "bad"}}) == 5


def test_prune_keeps_newest_numeric_checkpoints(tmp_path):
    for mins in (5, 10, 15):
        (tmp_path / f"checkpoint_{mins}m.pt").write_text("x")
    assert _prune_checkpoints(str(tmp_path), keep=2) == 3
    left = sorted(p.name for p in tmp_path.glob("checkpoint_*.pt"))
    assert left == ["checkpoint_10m.pt", "checkpoint_15m.pt"]
