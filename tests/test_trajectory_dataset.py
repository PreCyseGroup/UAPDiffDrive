import json
from pathlib import Path

import numpy as np

from uap_il.config import DeLucaGains, RobotParams, SimParams
from uap_il.dataset import (
    FEATURE_LABELS,
    POLICY_INPUT_DIM,
    add_state_collection_noise,
    add_wheel_action_noise,
    generate_demonstrations,
    sample_initial_position_offset,
    split_and_normalize,
)
from uap_il.trajectories import (
    TrajectorySpec,
    generate_lemniscate,
    generate_trajectory,
    reference_is_feasible,
    sample_trajectory_specs,
)


def test_lemniscate_shapes():
    traj = generate_lemniscate(
        1.0 / 60.0,
        TrajectorySpec(kind="lemniscate", duration=0.0, params={"eta": 1.0, "alpha": 4.0}),
    )
    assert len(traj.t) == len(traj.x) == len(traj.omega)
    assert np.all(np.isfinite(traj.v))
    np.testing.assert_allclose(traj.v, np.sqrt(traj.xd**2 + traj.yd**2), atol=1e-8)
    denom = np.maximum(traj.xd**2 + traj.yd**2, 1e-9)
    omega = (traj.ydd * traj.xd - traj.xdd * traj.yd) / denom
    np.testing.assert_allclose(traj.omega, omega, atol=1e-8)


def test_demonstrations_have_requested_features(tmp_path: Path):
    path = tmp_path / "demo.npz"
    data = generate_demonstrations(
        path,
        1,
        1,
        RobotParams(),
        SimParams(),
        DeLucaGains(),
        initial_xy_min_distance=0.10,
        initial_xy_max_distance=0.30,
        initial_theta_error=0.15,
    )
    assert path.exists()
    assert data["inputs"].shape[1] == POLICY_INPUT_DIM
    assert data["targets"].shape[1] == 2
    assert np.all(data["is_noisy_sample"] == 0)
    assert FEATURE_LABELS == [
        "ex",
        "ey",
        "theta",
        "xdr",
        "ydr",
        "xddr",
        "yddr",
        "vr",
    ]
    assert np.all(np.isfinite(data["inputs"][:, 2]))
    assert np.all(np.isfinite(data["inputs"][:, 3:]))
    assert np.all(data["inputs"][:, 7] >= 0.0)
    assert np.max(np.abs(data["targets"])) <= RobotParams().wheel_speed_limit
    first_idx = np.flatnonzero(data["run_ids"] == 0)[0]
    initial_distance = np.linalg.norm(data["states"][first_idx, :2] - data["refs"][first_idx, :2])
    assert 0.10 <= initial_distance <= 0.30


def test_initial_position_offset_distance_range():
    rng = np.random.default_rng(12)
    offsets = np.array([sample_initial_position_offset(rng) for _ in range(200)])
    distances = np.linalg.norm(offsets, axis=1)
    assert np.all(distances >= 0.10)
    assert np.all(distances <= 0.30)


def test_state_collection_noise_is_bounded_and_finite():
    rng = np.random.default_rng(7)
    state = np.array([1.0, -2.0, 3.12], dtype=np.float64)

    noisy = add_state_collection_noise(state, rng, xy_std=0.02, theta_std=0.05, clip_sigma=2.0)

    assert np.all(np.isfinite(noisy))
    assert np.max(np.abs(noisy[:2] - state[:2])) <= 0.04 + 1e-12
    assert -np.pi <= noisy[2] <= np.pi


def test_demonstration_metadata_records_state_noise(tmp_path: Path):
    path = tmp_path / "demo_noise.npz"
    data = generate_demonstrations(
        path,
        1,
        2,
        RobotParams(),
        SimParams(),
        DeLucaGains(),
        state_noise_xy_std=0.01,
        state_noise_theta_std=0.02,
        state_noise_clip_sigma=2.0,
        noisy_sample_fraction=0.5,
    )
    metadata = json.loads(str(data["metadata_json"]))

    assert metadata["data_collection"] == "expert_rollouts_with_state_noise_augmentation"
    assert metadata["state_noise"] == {
        "xy_std_m": 0.01,
        "theta_std_rad": 0.02,
        "clip_sigma": 2.0,
        "target_noisy_sample_fraction": 0.5,
        "noisy_sample_probability_per_clean_sample": 1.0,
    }
    assert np.any(data["is_noisy_sample"] == 1)
    assert np.any(data["is_noisy_sample"] == 0)


def test_wheel_action_noise_is_bounded():
    rng = np.random.default_rng(11)
    wr, wl = add_wheel_action_noise(9.9, -9.9, rng, RobotParams(), wheel_std=2.0, clip_sigma=1.0)

    assert -RobotParams().wheel_speed_limit <= wr <= RobotParams().wheel_speed_limit
    assert -RobotParams().wheel_speed_limit <= wl <= RobotParams().wheel_speed_limit


def test_dart_disturbs_only_configured_rollout_fraction(tmp_path: Path):
    path = tmp_path / "demo_dart.npz"
    data = generate_demonstrations(
        path,
        8,
        4,
        RobotParams(),
        SimParams(),
        DeLucaGains(),
        dart_disturbed_rollout_fraction=0.25,
        dart_wheel_noise_std=0.2,
    )
    metadata = json.loads(str(data["metadata_json"]))
    disturbed_ids = {
        run["run_id"]
        for run in metadata["runs"]
        if run["dart_disturbed"]
    }

    assert metadata["dart"]["disturbed_rollout_count"] == 2
    assert len(disturbed_ids) == 2
    assert set(np.unique(data["run_ids"][data["is_noisy_sample"] == 1])) == disturbed_ids


def test_dynamics_randomization_is_recorded_per_rollout(tmp_path: Path):
    path = tmp_path / "demo_domain_randomization.npz"
    data = generate_demonstrations(
        path,
        4,
        14,
        RobotParams(),
        SimParams(),
        DeLucaGains(),
        dynamics_randomization_fraction=0.5,
        wheel_radius_scale_range=(0.95, 1.05),
        wheelbase_scale_range=(0.95, 1.05),
        wheel_gain_range=(0.9, 1.1),
        motor_time_constant_range=(0.02, 0.08),
        action_delay_steps_range=(1, 2),
        observation_delay_steps_range=(0, 1),
    )
    metadata = json.loads(str(data["metadata_json"]))
    randomized = [run for run in metadata["runs"] if run["dynamics_randomized"]]

    assert metadata["dynamics_randomization"]["randomized_rollout_count"] == 2
    assert len(randomized) == 2
    assert all(1 <= run["dynamics"]["action_delay_steps"] <= 2 for run in randomized)
    assert all(0.02 <= run["dynamics"]["motor_time_constant_s"] <= 0.08 for run in randomized)
    assert np.all(np.isfinite(data["inputs"]))


def test_split_and_normalize_uses_disjoint_trajectory_ids():
    inputs = np.arange(40, dtype=np.float32).reshape(20, 2)
    targets = np.arange(40, 80, dtype=np.float32).reshape(20, 2)
    run_ids = np.repeat(np.arange(5), 4)

    stats = split_and_normalize(inputs, targets, seed=3, train_fraction=0.6, run_ids=run_ids)

    assert set(stats["train_run_ids"]).isdisjoint(set(stats["val_run_ids"]))
    assert set(np.unique(run_ids[stats["train_idx"]])) == set(stats["train_run_ids"])
    assert set(np.unique(run_ids[stats["val_idx"]])) == set(stats["val_run_ids"])
    assert len(stats["val_idx"]) > 0


def test_sampled_trajectory_families_are_mixed():
    specs = sample_trajectory_specs(70, 1, RobotParams(), 1.0 / 60.0)
    kinds = {spec.kind for spec in specs}
    assert {
        "lemniscate",
        "trig_combo",
        "interpolated",
        "circle",
        "ellipse",
        "lissajous",
        "rounded_box",
        "sine_lane",
        "slalom",
    }.issubset(kinds)


def test_circle_trajectory_is_finite_and_feasible():
    traj = generate_trajectory(
        1.0 / 60.0,
        TrajectorySpec(kind="circle", duration=58.0, params={"radius": 0.85, "phase": 0.35}),
    )
    assert traj.spec.kind == "circle"
    assert np.all(np.isfinite(traj.x))
    assert reference_is_feasible(traj, RobotParams())
