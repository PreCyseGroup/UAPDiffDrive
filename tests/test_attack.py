import numpy as np
import pytest
import torch

import uap_il.attack as attack_module
from uap_il.attack import build_policy_input_torch, fgsm_input_delta, learn_closed_loop_uap, learn_closed_loop_uap_gradient
from uap_il.cli import _build_showcase_eval_specs, _close_initial_state, _primary_attack_strategies, _worst_degraded_rollout_index
from uap_il.config import RobotParams, SimParams
from uap_il.dataset import POLICY_INPUT_DIM, tracking_features
from uap_il.model import WheelMLP
from uap_il.simulate import policy_wheels
from uap_il.trajectories import TrajectorySpec, generate_trajectory


def test_closed_loop_uap_respects_mask_and_epsilon():
    model = WheelMLP(input_dim=POLICY_INPUT_DIM, hidden_dim=8)
    stats = {
        "x_mean": np.zeros(POLICY_INPUT_DIM, dtype=np.float32),
        "x_std": np.ones(POLICY_INPUT_DIM, dtype=np.float32),
        "y_mean": np.zeros(2, dtype=np.float32),
        "y_std": np.ones(2, dtype=np.float32),
    }
    trajectory = generate_trajectory(
        0.1,
        TrajectorySpec(kind="circle", duration=1.0, params={"radius": 0.3, "phase": 0.0}),
    )
    mask = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    delta_raw, delta_norm, metadata = learn_closed_loop_uap(
        model,
        stats,
        [trajectory],
        [np.array([trajectory.x[0], trajectory.y[0], trajectory.theta[0]], dtype=np.float64)],
        RobotParams(),
        SimParams(dt=0.1, lookahead_time=0.1),
        epsilon=0.25,
        steps=1,
        population_size=4,
        seed=3,
        mask=mask,
        horizon_steps=5,
        score_mode="mean",
    )

    assert delta_raw.shape == (POLICY_INPUT_DIM,)
    assert delta_norm.shape == (POLICY_INPUT_DIM,)
    assert np.max(np.abs(delta_norm[:2])) <= 0.25 + 1e-6
    np.testing.assert_allclose(delta_norm[2:], 0.0, atol=1e-6)
    assert metadata["attack_method"] == "closed_loop_blackbox"
    assert metadata["evaluations"] >= 1


def test_closed_loop_gradient_uap_respects_mask_and_epsilon():
    model = WheelMLP(input_dim=POLICY_INPUT_DIM, hidden_dim=8)
    stats = {
        "x_mean": np.zeros(POLICY_INPUT_DIM, dtype=np.float32),
        "x_std": np.ones(POLICY_INPUT_DIM, dtype=np.float32),
        "y_mean": np.zeros(2, dtype=np.float32),
        "y_std": np.ones(2, dtype=np.float32),
    }
    trajectory = generate_trajectory(
        0.1,
        TrajectorySpec(kind="circle", duration=1.0, params={"radius": 0.3, "phase": 0.0}),
    )
    mask = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    delta_raw, delta_norm, metadata = learn_closed_loop_uap_gradient(
        model,
        stats,
        [trajectory],
        [np.array([trajectory.x[0], trajectory.y[0], trajectory.theta[0]], dtype=np.float64)],
        RobotParams(),
        SimParams(dt=0.1, lookahead_time=0.1),
        epsilon=0.25,
        steps=2,
        lr=0.05,
        mask=mask,
        horizon_steps=5,
        device="cpu",
    )

    assert delta_raw.shape == (POLICY_INPUT_DIM,)
    assert delta_norm.shape == (POLICY_INPUT_DIM,)
    assert np.max(np.abs(delta_norm[:2])) <= 0.25 + 1e-6
    np.testing.assert_allclose(delta_norm[2:], 0.0, atol=1e-6)
    assert metadata["attack_method"] == "closed_loop_gradient"
    assert metadata["optimizer"] == "Adam"
    assert metadata["final_attacked_rmse"] >= 0.0
    assert "score" in metadata
    assert "mean_rmse_pos_increase" in metadata


def test_closed_loop_gradient_stops_on_nonfinite_loss(monkeypatch):
    model = WheelMLP(input_dim=POLICY_INPUT_DIM, hidden_dim=8)
    stats = {
        "x_mean": np.zeros(POLICY_INPUT_DIM, dtype=np.float32),
        "x_std": np.ones(POLICY_INPUT_DIM, dtype=np.float32),
        "y_mean": np.zeros(2, dtype=np.float32),
        "y_std": np.ones(2, dtype=np.float32),
    }
    trajectory = generate_trajectory(
        0.1,
        TrajectorySpec(kind="circle", duration=1.0, params={"radius": 0.3, "phase": 0.0}),
    )
    mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    calls = {"count": 0}

    def fake_loss(model, stats_t, trajectory_t, initial_state, robot_params, sim_params, delta, horizon_steps=None):
        calls["count"] += 1
        if calls["count"] == 2:
            return delta.sum() * float("nan")
        return delta.sum() * 0.0 + 1.0

    monkeypatch.setattr(attack_module, "closed_loop_tracking_loss_torch", fake_loss)

    delta_raw, delta_norm, metadata = learn_closed_loop_uap_gradient(
        model,
        stats,
        [trajectory],
        [np.array([trajectory.x[0], trajectory.y[0], trajectory.theta[0]], dtype=np.float64)],
        RobotParams(),
        SimParams(dt=0.1, lookahead_time=0.1),
        epsilon=1.0,
        steps=3,
        lr=0.5,
        mask=mask,
        horizon_steps=5,
        device="cpu",
    )

    assert np.all(np.isfinite(delta_raw))
    assert np.all(np.isfinite(delta_norm))
    assert np.max(np.abs(delta_norm)) <= 1.0 + 1e-6
    assert metadata["stopped_early_reason"] == "nonfinite_loss"
    assert metadata["completed_steps"] == 1
    assert np.isfinite(metadata["final_attacked_rmse"])


def test_fgsm_input_delta_respects_xy_mask():
    model = WheelMLP(input_dim=POLICY_INPUT_DIM, hidden_dim=8)
    stats = {
        "x_mean": np.zeros(POLICY_INPUT_DIM, dtype=np.float32),
        "x_std": np.ones(POLICY_INPUT_DIM, dtype=np.float32),
        "y_mean": np.zeros(2, dtype=np.float32),
        "y_std": np.ones(2, dtype=np.float32),
    }
    feature = np.array([0.1, -0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.25], dtype=np.float32)
    mask = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    delta_raw, delta_norm = fgsm_input_delta(model, stats, feature, epsilon=0.5, mask=mask)

    assert delta_raw.shape == (POLICY_INPUT_DIM,)
    assert delta_norm.shape == (POLICY_INPUT_DIM,)
    assert np.max(np.abs(delta_norm[:2])) <= 0.5 + 1e-6
    np.testing.assert_allclose(delta_norm[2:], 0.0, atol=1e-6)


def test_torch_policy_input_matches_numpy_tracking_features():
    trajectory = generate_trajectory(
        0.1,
        TrajectorySpec(kind="circle", duration=1.0, params={"radius": 0.3, "phase": 0.0}),
    )
    state_np = np.array([0.1, -0.2, 0.4], dtype=np.float64)
    lookahead_steps = 1
    expected = tracking_features(state_np, trajectory, 2, lookahead_steps)
    trajectory_t = {
        "x": torch.as_tensor(trajectory.x, dtype=torch.float32),
        "y": torch.as_tensor(trajectory.y, dtype=torch.float32),
        "xd": torch.as_tensor(trajectory.xd, dtype=torch.float32),
        "yd": torch.as_tensor(trajectory.yd, dtype=torch.float32),
        "xdd": torch.as_tensor(trajectory.xdd, dtype=torch.float32),
        "ydd": torch.as_tensor(trajectory.ydd, dtype=torch.float32),
        "theta": torch.as_tensor(trajectory.theta, dtype=torch.float32),
        "v": torch.as_tensor(trajectory.v, dtype=torch.float32),
    }

    actual = build_policy_input_torch(torch.as_tensor(state_np, dtype=torch.float32), trajectory_t, 2, lookahead_steps)

    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)


def test_policy_wheels_rejects_stale_perturbation_dimension():
    model = WheelMLP(input_dim=POLICY_INPUT_DIM, hidden_dim=8)
    stats = {
        "x_mean": np.zeros(POLICY_INPUT_DIM, dtype=np.float32),
        "x_std": np.ones(POLICY_INPUT_DIM, dtype=np.float32),
        "y_mean": np.zeros(2, dtype=np.float32),
        "y_std": np.ones(2, dtype=np.float32),
    }
    feature = np.zeros(POLICY_INPUT_DIM, dtype=np.float32)
    stale_delta = np.zeros(POLICY_INPUT_DIM + 1, dtype=np.float32)

    with pytest.raises(ValueError, match="perturbation has shape"):
        policy_wheels(model, stats, feature, RobotParams(), perturbation=stale_delta)


def test_worst_degraded_rollout_index_uses_per_rollout_increase():
    clean = [{"rmse_pos": 5.0}, {"rmse_pos": 0.1}, {"rmse_pos": 0.2}]
    attacked = [{"rmse_pos": 4.0}, {"rmse_pos": 0.3}, {"rmse_pos": 0.25}]

    assert _worst_degraded_rollout_index(clean, attacked) == 1


def test_showcase_evaluation_is_lemniscate_only():
    specs = _build_showcase_eval_specs(seed=507, robot=RobotParams(), dt=0.1)

    assert len(specs) == 3
    assert {spec.kind for spec in specs} == {"lemniscate"}


def test_primary_attack_strategy_prefers_ex_ey_theta():
    strategies = {
        "xy_only": [1.0, 1.0] + [0.0] * (POLICY_INPUT_DIM - 2),
        "ex_ey_theta": [1.0, 1.0, 1.0] + [0.0] * (POLICY_INPUT_DIM - 3),
    }

    assert _primary_attack_strategies(strategies) == {"ex_ey_theta": strategies["ex_ey_theta"]}


def test_close_initial_state_uses_requested_offset_range():
    trajectory = generate_trajectory(
        0.1,
        TrajectorySpec(kind="circle", duration=1.0, params={"radius": 0.3, "phase": 0.0}),
    )
    rng = np.random.default_rng(4)

    initial = _close_initial_state(trajectory, rng, xy_error=0.30, theta_error=0.05)

    distance = np.linalg.norm(initial[:2] - np.array([trajectory.x[0], trajectory.y[0]]))
    assert 0.10 <= distance <= 0.30
