"""Unit tests for the SPRT champion-gate math (python/gai/sprt.py)."""
import pytest
from gai.sprt import sprt, elo_to_score, promote_decision


def test_elo_score_bounds_and_symmetry():
    assert abs(elo_to_score(0.0) - 0.5) < 1e-12
    assert elo_to_score(400.0) > 0.9
    assert elo_to_score(-400.0) < 0.1
    # symmetric: score(+x) + score(-x) == 1
    assert abs(elo_to_score(120.0) + elo_to_score(-120.0) - 1.0) < 1e-12


def test_continue_on_small_or_balanced_sample():
    assert sprt(1, 1, 0)[0] == "continue"
    assert sprt(100, 98, 4)[0] == "continue"


def test_accept_h1_on_dominant_candidate():
    dec, llr = sprt(300, 50, 0, elo0=0.0, elo1=5.0)
    assert dec == "accept_h1"
    assert llr > 0


def test_accept_h0_on_dominant_incumbent():
    dec, llr = sprt(50, 300, 0, elo0=0.0, elo1=5.0)
    assert dec == "accept_h0"
    assert llr < 0


def test_draws_count_as_half_win_half_loss():
    # 200 effective wins / 100 effective losses -> positive but not decisive vs elo1=5.
    dec, llr = sprt(100, 0, 200, elo0=0.0, elo1=5.0)
    assert dec == "continue"
    assert llr > 0


def test_llr_monotone_in_wins():
    _, lo = sprt(60, 40, 0)
    _, hi = sprt(90, 10, 0)
    assert hi > lo


def test_bad_parameters_raise():
    with pytest.raises(ValueError):
        sprt(10, 0, 0, elo0=5.0, elo1=5.0)
    with pytest.raises(ValueError):
        sprt(10, 0, 0, elo0=5.0, elo1=1.0)
    with pytest.raises(ValueError):
        sprt(10, 0, 0, alpha=0.0)
    with pytest.raises(ValueError):
        sprt(-1, 0, 0)


def test_promote_decision_mapping():
    assert promote_decision("accept_h1", {}) is True
    assert promote_decision("accept_h0", {}) is False
    assert promote_decision("max_games", {"w": 12, "l": 8}) is True
    assert promote_decision("max_games", {"w": 8, "l": 12}) is False
    assert promote_decision("max_games", {"w": 10, "l": 10}) is False  # tie keeps incumbent
