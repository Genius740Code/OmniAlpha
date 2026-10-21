"""Unit tests for the budget-clamped promotion gate and ordered endgame
(python/gai/orchestrator.py). All C++ calls are mocked — no build needed."""
import time
import pytest
import gai.orchestrator as orch


# ---- CHANGE 1: gate game plan converts the reserved window into games ----

def test_plan_gate_games_fits_window():
    ev = {"sprt_max_games": 200}
    games, cost = orch.plan_gate_games(ev, 360.0, 1.0)   # 6-min window, 1 min/game
    assert games == 6 and cost == pytest.approx(360.0)


def test_plan_gate_games_clamps_to_max_games():
    ev = {"sprt_max_games": 200}
    games, _ = orch.plan_gate_games(ev, 36000.0, 0.01)   # affordable 60000 -> cap 200
    assert games == 200


def test_plan_gate_games_no_budget_means_no_gate():
    assert orch.plan_gate_games({}, 0.0, 1.0) == (0, 0.0)
    assert orch.plan_gate_games({}, 100.0, 0.0) == (0, 0.0)


def test_plan_gate_games_ceil_can_round_up_but_stays_bounded():
    ev = {"sprt_max_games": 200}
    games, _ = orch.plan_gate_games(ev, 300.0, 4.0)      # 5-min window, 4 min/game
    assert games == 2


# ---- gate per-game cost estimate from measured selfplay speed ----

def test_estimate_gate_minutes_per_game_from_selfplay_speed():
    stats = {"games_per_s": 2.0, "sims": 100}
    # 0.5 s/game at 100 sims; gate at 200 sims with 1.5x safety -> 1.5 s = 0.025 min
    mpg = orch.estimate_gate_minutes_per_game(stats, {}, 200)
    assert mpg == pytest.approx(0.5 * 2.0 * 1.5 / 60.0)


def test_estimate_gate_minutes_per_game_fallback_constant():
    assert orch.estimate_gate_minutes_per_game(None, {}, 200) == pytest.approx(0.25)
    assert orch.estimate_gate_minutes_per_game({"games_per_s": 0}, {}, 200) == pytest.approx(0.25)
    assert orch.estimate_gate_minutes_per_game(None, {"gate_minutes_per_game": 2.0}, 200) == pytest.approx(2.0)


# ---- CHANGE 1: the gate never starts a batch that misses the deadline ----

def test_sprt_gate_aborts_when_deadline_already_passed(monkeypatch):
    calls = []

    def fake_sh_json(cmd):
        calls.append(cmd)
        return {"wins": 12, "losses": 8, "draws": 0}

    monkeypatch.setattr(orch, "sh_json", fake_sh_json)
    ev = {"sprt_batch_games": 20, "sprt_max_games": 200, "sprt_elo1": 35.0}
    dec, st = orch.sprt_arena_gate("build", "connect4", "nn:a", "nn:b", ev,
                                   max_games=200, deadline_s=time.time() - 1.0,
                                   minutes_per_game=1.0)
    assert calls == []
    assert dec == "max_games" and st["budget_aborted"] is True and st["games"] == 0


def test_sprt_gate_trims_batches_to_fit_deadline(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(orch.time, "time", lambda: now[0])

    def fake_sh_json(cmd):
        i = cmd.index("--games")
        n = int(cmd[i + 1])
        now[0] += n * 30.0  # games run 2x faster than the 1 min/game estimate
        return {"wins": n // 2, "losses": n - n // 2, "draws": 0}

    monkeypatch.setattr(orch, "sh_json", fake_sh_json)
    ev = {"sprt_batch_games": 20, "sprt_max_games": 200, "sprt_elo1": 35.0}
    # 600 s of budget at an estimated 1 min/game: the first batch is trimmed to 10
    # (not 20), then batches keep shrinking 5,2,1,1 until the window is exhausted.
    dec, st = orch.sprt_arena_gate("build", "connect4", "nn:a", "nn:b", ev,
                                   max_games=200, deadline_s=1600.0,
                                   minutes_per_game=1.0)
    assert st["budget_aborted"] is True
    assert st["games"] == 19
    assert dec == "max_games"


def test_sprt_gate_resolves_before_deadline(monkeypatch):
    monkeypatch.setattr(orch.time, "time", lambda: 1000.0)

    def fake_sh_json(cmd):
        return {"wins": 18, "losses": 2, "draws": 0}  # dominant candidate

    monkeypatch.setattr(orch, "sh_json", fake_sh_json)
    ev = {"sprt_batch_games": 20, "sprt_max_games": 200, "sprt_elo1": 35.0}
    dec, st = orch.sprt_arena_gate("build", "connect4", "nn:a", "nn:b", ev,
                                   max_games=200, deadline_s=2000.0,
                                   minutes_per_game=1.0)
    assert dec == "accept_h1" and st["budget_aborted"] is False
    assert st["games"] == 40  # resolved after 2 batches, not 200


# ---- CHANGE 1: skip-gate path prunes checkpoints, keeps the newest 5 ----

def test_prune_checkpoints_keeps_newest_five(tmp_path):
    for m in range(1, 8):
        (tmp_path / f"checkpoint_{m}m.pt").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")
    kept = orch._prune_checkpoints(str(tmp_path), keep=5)
    assert kept == 7
    remaining = sorted(int(p.name.split("_")[1].split("m")[0])
                       for p in tmp_path.glob("checkpoint_*.pt"))
    assert remaining == [3, 4, 5, 6, 7]
    assert (tmp_path / "latest.pt").exists()  # never touch latest


# ---- CHANGE 2: equal-time flags are forwarded only when configured ----

def test_eval_extra_flags_off_by_default():
    assert orch._eval_extra_flags({}) == []
    assert orch._eval_extra_flags({"tc_ms": None}) == []


def test_eval_extra_flags_forwarded_when_set():
    flags = orch._eval_extra_flags({"tc_ms": 1000, "tc_a": 500, "tc_b": 250})
    assert flags == ["--tc-ms", "1000", "--tc-a", "500", "--tc-b", "250"]


def test_eval_extra_flags_ignore_sub_20ms_and_garbage():
    assert orch._eval_extra_flags({"tc_ms": 10}) == []
    assert orch._eval_extra_flags({"tc_ms": "abc"}) == []
    assert orch._eval_extra_flags({"tc_a": 5, "tc_b": 100}) == ["--tc-b", "100"]


# ---- CHANGE 2: ordered endgame tournament ranks via Bradley-Terry ----

def test_endgame_tournament_picks_bt_winner(monkeypatch, tmp_path):
    # Seed candidates with pre-existing (dummy) .ts files so no export is needed.
    (tmp_path / "latest.ts").write_bytes(b"x")
    (tmp_path / "champion.ts").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")

    def fake_sh_json(cmd):
        # candidate (evaluator-a, listed first) beats champion 75-25
        if cmd[cmd.index("--evaluator-a") + 1].endswith("latest.ts"):
            return {"wins": 15, "losses": 5, "draws": 0}
        return {"wins": 5, "losses": 15, "draws": 0}

    monkeypatch.setattr(orch, "sh_json", fake_sh_json)
    cfg = {"network": {"type": "mlp", "hidden": 8, "layers": 1}}
    shape = {"planes": 1, "h": 1, "w": 1, "actions": 2}
    ev = {"roundrobin_last": 5, "roundrobin_games": 20, "sprt_batch_games": 20}
    res = orch.run_endgame_tournament("build", "connect4", cfg, str(tmp_path), shape, ev,
                                      deadline_s=time.time() + 3600.0,
                                      minutes_per_game=0.01)
    assert res is not None
    assert res["winner"] == "latest"
    assert res["elo"]["latest"] > res["elo"]["champion"]
    assert res["pairs"][(0, 1)] == (15, 5, 0)


def test_endgame_tournament_needs_two_candidates(monkeypatch, tmp_path):
    (tmp_path / "latest.ts").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")
    cfg = {"network": {"type": "mlp", "hidden": 8, "layers": 1}}
    shape = {"planes": 1, "h": 1, "w": 1, "actions": 2}
    res = orch.run_endgame_tournament("build", "connect4", cfg, str(tmp_path), shape, {},
                                      deadline_s=time.time() + 3600.0,
                                      minutes_per_game=0.01)
    assert res is None


def test_endgame_tournament_skips_when_budget_too_small(monkeypatch, tmp_path):
    (tmp_path / "latest.ts").write_bytes(b"x")
    (tmp_path / "champion.ts").write_bytes(b"x")
    (tmp_path / "latest.pt").write_bytes(b"x")
    monkeypatch.setattr(orch, "sh_json", lambda cmd: {"wins": 1, "losses": 1, "draws": 0})
    cfg = {"network": {"type": "mlp", "hidden": 8, "layers": 1}}
    shape = {"planes": 1, "h": 1, "w": 1, "actions": 2}
    # 1 pair, 10% of 10 s at 1 min/game -> 0 games/pair -> not feasible
    res = orch.run_endgame_tournament("build", "connect4", cfg, str(tmp_path), shape, {},
                                      deadline_s=time.time() + 10.0,
                                      minutes_per_game=1.0)
    assert res is None
