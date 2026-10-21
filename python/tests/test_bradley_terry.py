"""Bradley-Terry Elo fit used by the ordered endgame tournament (python/gai/sprt.py)."""
import math
from gai.sprt import bradley_terry_elo


def test_two_players_elo_difference_matches_score():
    # 75-25 over 100 games -> 400*log10(3) ~ 190.8 Elo apart; field centred at 0.
    elo = bradley_terry_elo(["a", "b"], {(0, 1): (75, 25, 0)})
    assert abs((elo["a"] - elo["b"]) - 400.0 * math.log10(3.0)) < 0.5
    assert abs(elo["a"] + elo["b"]) < 1e-6


def test_three_players_transitive_ordering():
    names = ["a", "b", "c"]
    pair = {(0, 1): (60, 40, 0), (0, 2): (70, 30, 0), (1, 2): (55, 45, 0)}
    elo = bradley_terry_elo(names, pair)
    assert elo["a"] > elo["b"] > elo["c"]
    # played field is centred: elos sum to 0
    assert abs(sum(elo.values())) < 1e-6


def test_draws_pull_players_together():
    elo = bradley_terry_elo(["x", "y"], {(0, 1): (0, 0, 10)})
    assert abs(elo["x"]) < 1e-6 and abs(elo["y"]) < 1e-6
    # 60% expected score (2 wins + 8 draws) vs 50% -> small positive gap (~70 Elo)
    elo = bradley_terry_elo(["x", "y"], {(0, 1): (2, 0, 8)})
    assert 0.0 < elo["x"] - elo["y"] < 100.0


def test_unplayed_player_gets_zero_and_does_not_distort():
    elo = bradley_terry_elo(["a", "b", "loner"], {(0, 1): (75, 25, 0)})
    assert elo["loner"] == 0.0
    assert abs((elo["a"] - elo["b"]) - 400.0 * math.log10(3.0)) < 0.5


def test_winless_player_stays_finite():
    elo = bradley_terry_elo(["a", "b"], {(0, 1): (100, 0, 0)})
    assert math.isfinite(elo["a"]) and math.isfinite(elo["b"])
    assert elo["b"] < elo["a"]


def test_empty_table_gives_all_zero():
    elo = bradley_terry_elo(["a", "b"], {})
    assert elo == {"a": 0.0, "b": 0.0}
