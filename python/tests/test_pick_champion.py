"""Unit tests for champion-pick ranking math (python/gai/pick_champion.py)."""
from gai.pick_champion import rank_candidates, score_to_elo


def test_rank_candidates_orders_by_score():
    names = ["a", "b", "c"]
    # a beats b and c; b beats c
    pair = {(0, 1): (60, 40, 0), (0, 2): (70, 30, 0), (1, 2): (55, 45, 0)}
    rows = rank_candidates(names, pair)
    assert [r[0] for r in rows] == ["a", "b", "c"]
    # a: 60+70 wins, 40+30 losses -> 130-70
    assert (rows[0][1], rows[0][2]) == (130, 70)
    assert rows[0][4] > rows[1][4] > rows[2][4]


def test_rank_candidates_draws_count_half():
    names = ["x", "y"]
    rows = rank_candidates(names, {(0, 1): (0, 0, 10)})
    assert abs(rows[0][4] - 0.5) < 1e-12
    assert abs(rows[1][4] - 0.5) < 1e-12


def test_score_to_elo_anchors():
    assert abs(score_to_elo(0.5)) < 1e-9
    assert score_to_elo(0.75) > 150  # ~+190 Elo
    assert score_to_elo(0.25) < -150
