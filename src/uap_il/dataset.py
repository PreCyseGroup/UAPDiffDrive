from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import DeLucaGains, RobotParams, SimParams
from .controllers import ControllerState, deluca_controller
from .robot import step_differential_drive, wheel_to_unicycle, wrap_to_pi
from .trajectories import generate_trajectory, sample_trajectory_specs

FEATURE_LABELS = [
    "ex",
    "ey",
    "theta",
    "xdr",
    "ydr",
    "xddr",
    "yddr",
    "vr",
]
POLICY_INPUT_DIM = len(FEATURE_LABELS)


def sample_initial_position_offset(
    rng: np.random.Generator,
    min_distance: float = 0.10,
    max_distance: float = 0.30,
) -> np.ndarray:
    """Sample an xy offset whose Euclidean distance is within the requested range."""
    if min_distance < 0.0:
        raise ValueError("min_distance must be non-negative.")
    if max_distance < min_distance:
        raise ValueError("max_distance must be greater than or equal to min_distance.")
    distance = rng.uniform(min_distance, max_distance)
    angle = rng.uniform(-np.pi, np.pi)
    return np.array([distance * np.cos(angle), distance * np.sin(angle)], dtype=np.float64)


def add_state_collection_noise(
    state: np.ndarray,
    rng: np.random.Generator,
    xy_std: float = 0.0,
    theta_std: float = 0.0,
    clip_sigma: float = 3.0,
) -> np.ndarray:
    """Return a noisy state sample for collecting local recovery examples."""
    noisy = state.astype(np.float64).copy()
    if xy_std > 0.0:
        xy_noise = rng.normal(0.0, xy_std, size=2)
        if clip_sigma > 0.0:
            xy_noise = np.clip(xy_noise, -clip_sigma * xy_std, clip_sigma * xy_std)
        noisy[:2] += xy_noise
    if theta_std > 0.0:
        theta_noise = float(rng.normal(0.0, theta_std))
        if clip_sigma > 0.0:
            theta_noise = float(np.clip(theta_noise, -clip_sigma * theta_std, clip_sigma * theta_std))
        noisy[2] = wrap_to_pi(noisy[2] + theta_noise)
    return noisy


def add_wheel_action_noise(
    wr: float,
    wl: float,
    rng: np.random.Generator,
    robot_params: RobotParams,
    wheel_std: float = 0.0,
    clip_sigma: float = 2.0,
) -> tuple[float, float]:
    """Perturb an expert wheel command for DART-style disturbed rollouts."""
    if wheel_std <= 0.0:
        return wr, wl
    noise = rng.normal(0.0, wheel_std, size=2)
    if clip_sigma > 0.0:
        noise = np.clip(noise, -clip_sigma * wheel_std, clip_sigma * wheel_std)
    wr_noisy = float(np.clip(wr + noise[0], -robot_params.wheel_speed_limit, robot_params.wheel_speed_limit))
    wl_noisy = float(np.clip(wl + noise[1], -robot_params.wheel_speed_limit, robot_params.wheel_speed_limit))
    return wr_noisy, wl_noisy


def unwrap_heading_to_reference(theta: float | np.ndarray, reference_theta: float | np.ndarray) -> float | np.ndarray:
    """Represent a wrapped robot heading on the same continuous branch as the reference heading."""
    return reference_theta + wrap_to_pi(theta - reference_theta)


def tracking_features(state: np.ndarray, trajectory, k: int, lookahead_steps: int) -> np.ndarray:
    feature_k = min(k + lookahead_steps, len(trajectory.t) - 1)
    dx = trajectory.x[feature_k] - state[0]
    dy = trajectory.y[feature_k] - state[1]
    ex = np.cos(state[2]) * dx + np.sin(state[2]) * dy
    ey = -np.sin(state[2]) * dx + np.cos(state[2]) * dy
    theta = unwrap_heading_to_reference(state[2], trajectory.theta[k])
    return np.array(
        [
            ex,
            ey,
            theta,
            trajectory.xd[k],
            trajectory.yd[k],
            trajectory.xdd[k],
            trajectory.ydd[k],
            trajectory.v[k],
        ],
        dtype=np.float32,
    )


def generate_demonstrations(
    output_path: Path,
    num_runs: int,
    seed: int,
    robot_params: RobotParams,
    sim_params: SimParams,
    gains: DeLucaGains,
    state_noise_xy_std: float = 0.0,
    state_noise_theta_std: float = 0.0,
    state_noise_clip_sigma: float = 3.0,
    noisy_sample_fraction: float = 0.0,
    dart_disturbed_rollout_fraction: float = 0.0,
    dart_wheel_noise_std: float = 0.0,
    dart_action_clip_sigma: float = 2.0,
    initial_xy_min_distance: float = 0.05,
    initial_xy_max_distance: float = 0.40,
    initial_theta_error: float = 0.25,
    dynamics_randomization_fraction: float = 0.0,
    wheel_radius_scale_range: tuple[float, float] = (1.0, 1.0),
    wheelbase_scale_range: tuple[float, float] = (1.0, 1.0),
    wheel_gain_range: tuple[float, float] = (1.0, 1.0),
    motor_time_constant_range: tuple[float, float] = (0.0, 0.0),
    action_delay_steps_range: tuple[int, int] = (0, 0),
    observation_delay_steps_range: tuple[int, int] = (0, 0),
    trajectory_family_order: list[str] | None = None,
    target_mean_abs_wheel_range: tuple[float, float] = (5.8, 8.8),
    min_reference_peak: float = 8.4,
    max_reference_peak_fraction: float = 0.995,
    use_robot_velocity_feature: bool = False,
    log_progress: bool = False,
) -> dict[str, np.ndarray]:
    if not 0.0 <= noisy_sample_fraction <= 0.5:
        raise ValueError("noisy_sample_fraction must be in [0.0, 0.5].")
    if not 0.0 <= dart_disturbed_rollout_fraction <= 0.5:
        raise ValueError("dart_disturbed_rollout_fraction must be in [0.0, 0.5].")
    if not 0.0 <= dynamics_randomization_fraction <= 1.0:
        raise ValueError("dynamics_randomization_fraction must be in [0.0, 1.0].")
    for name, bounds in {
        "wheel_radius_scale_range": wheel_radius_scale_range,
        "wheelbase_scale_range": wheelbase_scale_range,
        "wheel_gain_range": wheel_gain_range,
        "motor_time_constant_range": motor_time_constant_range,
    }.items():
        if len(bounds) != 2 or bounds[0] > bounds[1] or bounds[0] < 0.0:
            raise ValueError(f"{name} must be an ordered non-negative pair.")
    for name, bounds in {
        "action_delay_steps_range": action_delay_steps_range,
        "observation_delay_steps_range": observation_delay_steps_range,
    }.items():
        if len(bounds) != 2 or bounds[0] > bounds[1] or bounds[0] < 0:
            raise ValueError(f"{name} must be an ordered non-negative integer pair.")
    rng = np.random.default_rng(seed + 17)
    if log_progress:
        print(f"[data] Sampling {num_runs} feasible trajectory specs", flush=True)
    specs = sample_trajectory_specs(
        num_runs,
        seed,
        robot_params,
        sim_params.dt,
        family_order=trajectory_family_order,
        target_mean_abs_wheel_range=target_mean_abs_wheel_range,
        min_reference_peak=min_reference_peak,
        max_reference_peak_fraction=max_reference_peak_fraction,
    )
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    run_ids: list[np.ndarray] = []
    noisy_flags: list[np.ndarray] = []
    meta_runs = []
    noisy_sample_probability = (
        noisy_sample_fraction / (1.0 - noisy_sample_fraction) if noisy_sample_fraction > 0.0 else 0.0
    )
    noise_enabled = (state_noise_xy_std > 0.0 or state_noise_theta_std > 0.0) and noisy_sample_probability > 0.0
    dart_enabled = dart_wheel_noise_std > 0.0 and dart_disturbed_rollout_fraction > 0.0
    disturbed_run_ids: set[int] = set()
    if dart_enabled:
        disturbed_count = max(1, int(round(dart_disturbed_rollout_fraction * num_runs)))
        disturbed_count = min(disturbed_count, num_runs)
        disturbed_run_ids = set(rng.choice(num_runs, size=disturbed_count, replace=False).tolist())
    randomized_run_ids: set[int] = set()
    if dynamics_randomization_fraction > 0.0:
        randomized_count = max(1, int(round(dynamics_randomization_fraction * num_runs)))
        randomized_run_ids = set(rng.choice(num_runs, size=min(randomized_count, num_runs), replace=False).tolist())

    for run_id, spec in enumerate(specs):
        traj = generate_trajectory(sim_params.dt, spec)
        disturb_rollout = run_id in disturbed_run_ids
        randomize_dynamics = run_id in randomized_run_ids
        radius_scale = float(rng.uniform(*wheel_radius_scale_range)) if randomize_dynamics else 1.0
        base_scale = float(rng.uniform(*wheelbase_scale_range)) if randomize_dynamics else 1.0
        right_gain = float(rng.uniform(*wheel_gain_range)) if randomize_dynamics else 1.0
        left_gain = float(rng.uniform(*wheel_gain_range)) if randomize_dynamics else 1.0
        motor_tau = float(rng.uniform(*motor_time_constant_range)) if randomize_dynamics else 0.0
        action_delay = int(rng.integers(action_delay_steps_range[0], action_delay_steps_range[1] + 1)) if randomize_dynamics else 0
        observation_delay = int(rng.integers(observation_delay_steps_range[0], observation_delay_steps_range[1] + 1)) if randomize_dynamics else 0
        rollout_robot = RobotParams(
            wheel_radius=robot_params.wheel_radius * radius_scale,
            wheelbase=robot_params.wheelbase * base_scale,
            wheel_speed_limit=robot_params.wheel_speed_limit,
        )
        if log_progress:
            print(
                f"[data] rollout {run_id + 1:03d}/{num_runs:03d} "
                f"kind={spec.kind} steps={len(traj.t)} disturbed={disturb_rollout} randomized={randomize_dynamics}",
                flush=True,
            )
        lookahead_steps = max(0, int(round(sim_params.lookahead_time / sim_params.dt)))
        xy_offset = sample_initial_position_offset(
            rng,
            min_distance=initial_xy_min_distance,
            max_distance=initial_xy_max_distance,
        )
        state = np.array(
            [
                traj.x[0] + xy_offset[0],
                traj.y[0] + xy_offset[1],
                traj.theta[0] + rng.uniform(-initial_theta_error, initial_theta_error),
            ],
            dtype=np.float64,
        )
        ctrl_state = ControllerState(v_curr=float(traj.v[0]))
        state_history = [state.copy()]
        command_queue = [(0.0, 0.0)] * action_delay
        applied_wr = 0.0
        applied_wl = 0.0
        for k in range(len(traj.t)):
            observed_state = state_history[max(0, len(state_history) - 1 - observation_delay)]
            feature = tracking_features(observed_state, traj, k, lookahead_steps)
            if use_robot_velocity_feature:
                feature[-1] = np.float32(ctrl_state.v_curr)
            wr, wl, _ = deluca_controller(observed_state, k, traj, ctrl_state, gains, robot_params, sim_params.dt)
            inputs.append(feature)
            targets.append(np.array([wr, wl], dtype=np.float32))
            states.append(observed_state.astype(np.float32))
            refs.append(np.array([traj.x[k], traj.y[k], traj.theta[k]], dtype=np.float32))
            run_ids.append(np.array([run_id], dtype=np.int32))
            noisy_flags.append(np.array([1 if disturb_rollout else 0], dtype=np.int8))

            if noise_enabled and rng.random() < noisy_sample_probability:
                noisy_state = add_state_collection_noise(
                    observed_state,
                    rng,
                    xy_std=state_noise_xy_std,
                    theta_std=state_noise_theta_std,
                    clip_sigma=state_noise_clip_sigma,
                )
                noisy_feature = tracking_features(noisy_state, traj, k, lookahead_steps)
                if use_robot_velocity_feature:
                    noisy_feature[-1] = np.float32(ctrl_state.v_curr)
                noisy_wr, noisy_wl, _ = deluca_controller(
                    noisy_state, k, traj, ctrl_state, gains, robot_params, sim_params.dt
                )
                inputs.append(noisy_feature)
                targets.append(np.array([noisy_wr, noisy_wl], dtype=np.float32))
                states.append(noisy_state.astype(np.float32))
                refs.append(np.array([traj.x[k], traj.y[k], traj.theta[k]], dtype=np.float32))
                run_ids.append(np.array([run_id], dtype=np.int32))
                noisy_flags.append(np.array([1], dtype=np.int8))

            exec_wr, exec_wl = (
                add_wheel_action_noise(
                    wr,
                    wl,
                    rng,
                    robot_params,
                    wheel_std=dart_wheel_noise_std,
                    clip_sigma=dart_action_clip_sigma,
                )
                if disturb_rollout
                else (wr, wl)
            )
            command_queue.append((exec_wr, exec_wl))
            delayed_wr, delayed_wl = command_queue.pop(0)
            if motor_tau > 0.0:
                motor_alpha = 1.0 - np.exp(-sim_params.dt / motor_tau)
                applied_wr += motor_alpha * (delayed_wr - applied_wr)
                applied_wl += motor_alpha * (delayed_wl - applied_wl)
            else:
                applied_wr, applied_wl = delayed_wr, delayed_wl
            physical_wr = right_gain * applied_wr
            physical_wl = left_gain * applied_wl
            state = step_differential_drive(state, physical_wr, physical_wl, sim_params.dt, rollout_robot)
            state_history.append(state.copy())
            executed_v, _ = wheel_to_unicycle(physical_wr, physical_wl, rollout_robot)
            ctrl_state = ControllerState(v_curr=executed_v)
        meta_runs.append(
            {
                "run_id": run_id,
                "spec": asdict(spec),
                "num_steps": len(traj.t),
                "dart_disturbed": disturb_rollout,
                "dynamics_randomized": randomize_dynamics,
                "dynamics": {
                    "wheel_radius_scale": radius_scale,
                    "wheelbase_scale": base_scale,
                    "right_wheel_gain": right_gain,
                    "left_wheel_gain": left_gain,
                    "motor_time_constant_s": motor_tau,
                    "action_delay_steps": action_delay,
                    "observation_delay_steps": observation_delay,
                },
            }
        )

    if log_progress:
        print(f"[data] Writing {len(inputs)} samples -> {output_path}", flush=True)
    data = {
        "inputs": np.vstack(inputs),
        "targets": np.vstack(targets),
        "states": np.vstack(states),
        "refs": np.vstack(refs),
        "run_ids": np.vstack(run_ids).reshape(-1),
        "is_noisy_sample": np.vstack(noisy_flags).reshape(-1),
        "metadata_json": np.array(
            json.dumps(
                {
                    "robot_params": asdict(robot_params),
                    "sim_params": asdict(sim_params),
                    "deluca_gains": asdict(gains),
                    "feature_labels": (
                        [*FEATURE_LABELS[:-1], "v_robot"] if use_robot_velocity_feature else FEATURE_LABELS
                    ),
                    "feature_contract": (
                        "legacy-theta-vrobot-deluca"
                        if use_robot_velocity_feature
                        else {
                            "velocity_feature": FEATURE_LABELS[-1],
                            "v_robot_semantics": None,
                        }
                    ),
                    "feature_contract_details": {
                        "velocity_feature": "v_robot" if use_robot_velocity_feature else FEATURE_LABELS[-1],
                        "v_robot_semantics": (
                            "actual forward velocity from the previously executed physical wheel speeds"
                            if use_robot_velocity_feature
                            else None
                        ),
                    },
                    "expert": {
                        "name": "De Luca dynamic-feedback linearization controller",
                        "implementation": "src/uap_il/controllers.py::deluca_controller",
                        "gains": asdict(gains),
                        "label_policy": "clean expert action at each visited state",
                    },
                    "data_collection": "expert_rollouts_with_state_noise_augmentation",
                    "initial_xy_distance_range_m": [initial_xy_min_distance, initial_xy_max_distance],
                    "initial_theta_error_range": [-initial_theta_error, initial_theta_error],
                    "state_noise": {
                        "xy_std_m": state_noise_xy_std,
                        "theta_std_rad": state_noise_theta_std,
                        "clip_sigma": state_noise_clip_sigma,
                        "target_noisy_sample_fraction": noisy_sample_fraction,
                        "noisy_sample_probability_per_clean_sample": noisy_sample_probability,
                    },
                    "dart": {
                        "disturbed_rollout_fraction": dart_disturbed_rollout_fraction,
                        "disturbed_rollout_count": len(disturbed_run_ids),
                        "wheel_noise_std_rad_s": dart_wheel_noise_std,
                        "action_clip_sigma": dart_action_clip_sigma,
                        "label_policy": "clean_expert_action_at_visited_state",
                    },
                    "dynamics_randomization": {
                        "randomized_rollout_fraction": dynamics_randomization_fraction,
                        "randomized_rollout_count": len(randomized_run_ids),
                        "wheel_radius_scale_range": list(wheel_radius_scale_range),
                        "wheelbase_scale_range": list(wheelbase_scale_range),
                        "wheel_gain_range": list(wheel_gain_range),
                        "motor_time_constant_range_s": list(motor_time_constant_range),
                        "action_delay_steps_range": list(action_delay_steps_range),
                        "observation_delay_steps_range": list(observation_delay_steps_range),
                    },
                    "trajectory_sampling": {
                        "family_order": trajectory_family_order,
                        "target_mean_abs_wheel_range": list(target_mean_abs_wheel_range),
                        "min_reference_peak": min_reference_peak,
                        "max_reference_peak_fraction": max_reference_peak_fraction,
                    },
                    "runs": meta_runs,
                },
                indent=2,
            )
        ),
    }
    np.savez_compressed(output_path, **data)
    return data


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


def split_and_normalize(
    inputs: np.ndarray,
    targets: np.ndarray,
    seed: int,
    train_fraction: float = 0.85,
    run_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if run_ids is None:
        indices = np.arange(len(inputs))
        rng.shuffle(indices)
        n_train = int(train_fraction * len(indices))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        train_runs = np.array([], dtype=np.int32)
        val_runs = np.array([], dtype=np.int32)
    else:
        run_ids = np.asarray(run_ids)
        unique_runs = np.unique(run_ids)
        rng.shuffle(unique_runs)
        n_train_runs = max(1, int(round(train_fraction * len(unique_runs))))
        n_train_runs = min(n_train_runs, len(unique_runs) - 1) if len(unique_runs) > 1 else len(unique_runs)
        train_runs = np.sort(unique_runs[:n_train_runs])
        val_runs = np.sort(unique_runs[n_train_runs:])
        train_idx = np.flatnonzero(np.isin(run_ids, train_runs))
        val_idx = np.flatnonzero(np.isin(run_ids, val_runs))
        if len(val_idx) == 0:
            raise ValueError("Trajectory-level validation split needs at least two runs.")

    x_mean = inputs[train_idx].mean(axis=0)
    x_std = inputs[train_idx].std(axis=0) + 1e-6
    y_mean = targets[train_idx].mean(axis=0)
    y_std = targets[train_idx].std(axis=0) + 1e-6

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_run_ids": train_runs.astype(np.int32),
        "val_run_ids": val_runs.astype(np.int32),
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean.astype(np.float32),
        "y_std": y_std.astype(np.float32),
    }
