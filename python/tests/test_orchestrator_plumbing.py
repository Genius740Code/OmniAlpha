"""Orchestrator plumbing fixes: per-game NN probe + steps/iter clamp.

1. build_supports_nn used to probe `--game connect4` no matter what game was
   being trained; the probe answers for the wrong game and can flip the
   evaluator-switch decision for every other game.
2. compute_effective_steps is the extracted v9-starvation clamp; the unit test
   pins the denominator semantics (post-split, post-augment fresh positions)
   that once starved training 11.3 -> 8 steps/iter.
"""
import gai.orchestrator as orch


class _Proc:
    def __init__(self, rc):
        self.returncode = rc


def test_build_supports_nn_probes_the_actual_game(monkeypatch):
    seen = []

    def fake_run(cmd):
        seen.append(cmd)
        return _Proc(0)

    monkeypatch.setattr(orch, "run_cmd", fake_run)
    assert orch.build_supports_nn("some_build", "breakthrough") is True
    cmd = seen[0]
    assert "--game" in cmd and cmd[cmd.index("--game") + 1] == "breakthrough"
    assert "connect4" not in cmd
    assert cmd[0] == "some_build/selfplay"
    # the probe file lives in the approved scratch dir, not repo cwd
    assert any(str(a).startswith("/tmp/opencode/") for a in cmd)


def test_build_supports_nn_maps_returncode(monkeypatch):
    monkeypatch.setattr(orch, "run_cmd", lambda cmd: _Proc(1))
    assert orch.build_supports_nn("build", "connect4") is False
    monkeypatch.setattr(orch, "run_cmd", lambda cmd: _Proc(0))
    assert orch.build_supports_nn("build", "connect4") is True


def test_steps_healthy_regime_not_clamped():
    # recent real runs: ~990 fresh/iter, batch 32, mrr 3 -> clamp 92 >> steps_per_iter
    assert orch.compute_effective_steps(13, 990, 32, 3.0, 4096) == 13


def test_steps_clamp_binds_on_small_batches():
    assert orch.compute_effective_steps(30, 100, 32, 3.0, 4096) == 9


def test_steps_denominator_is_post_augment_positions():
    # v9 regression shape: augmenting roughly doubles the fresh-position count,
    # and the clamp must scale WITH it (more data -> more steps allowed).
    # steps_per_iter=50 is high enough that the clamp binds in both cases.
    without_augment = orch.compute_effective_steps(50, 250, 32, 3.0, 4096)
    with_augment = orch.compute_effective_steps(50, 500, 32, 3.0, 4096)
    assert (without_augment, with_augment) == (23, 46)  # 2x data -> 2x steps
    # the starvation direction: an undercounting denominator gives FEWER steps
    assert orch.compute_effective_steps(30, 100, 32, 3.0, 4096) < \
           orch.compute_effective_steps(30, 200, 32, 3.0, 4096)


def test_steps_guard_trains_one_step_when_buffer_can_fill():
    # fresh positions can't fill half a batch, but the buffer already holds
    # >= batch: must still train 1 step, never 0 (silent stall).
    assert orch.compute_effective_steps(10, 10, 32, 3.0, 64) == 1


def test_steps_zero_when_buffer_cannot_fill_batch():
    assert orch.compute_effective_steps(10, 10, 32, 3.0, 16) == 0


def test_steps_truncates_not_rounds():
    # int(3*11/32) == 1 (truncation), not round(1.03) == 1... same here, so
    # pick a case that separates them: int(3*21/32)=1 vs round(1.97)=2.
    assert orch.compute_effective_steps(10, 21, 32, 3.0, 4096) == 1
