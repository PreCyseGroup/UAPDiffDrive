from __future__ import annotations

import numpy as np
import torch

from .config import RobotParams, SimParams
from .dataset import tracking_features
from .robot import clip_wheels, step_differential_drive, wrap_to_pi
from .trajectories import Trajectory


def _validate_policy_vector(name: str, value: np.ndarray, expected_dim: int) -> np.ndarray:
    vector = value.astype(np.float32)
    if vector.shape != (expected_dim,):
        raise ValueError(f"{name} has shape {vector.shape}, expected ({expected_dim},).")
    return vector


def policy_wheels(
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    feature: np.ndarray,
    robot_params: RobotParams,
    perturbation: np.ndarray | None = None,
) -> tuple[float, float]:
    input_dim = int(stats["x_mean"].shape[0])
    x_raw = _validate_policy_vector("feature", feature, input_dim)
    if perturbation is not None:
        x_raw = x_raw + _validate_policy_vector("perturbation", perturbation, input_dim)
    x = (x_raw - stats["x_mean"]) / stats["x_std"]
    device = next(model.parameters()).device
    with torch.no_grad():
        x_tensor = torch.from_numpy(x[None, :]).to(device)
        y_norm = model(x_tensor).detach().cpu().numpy()[0]
    wr, wl = y_norm * stats["y_std"] + stats["y_mean"]
    return clip_wheels(float(wr), float(wl), robot_params)


def rollout_policy(
    model: torch.nn.Module,
    stats: dict[str, np.ndarray],
    trajectory: Trajectory,
    initial_state: np.ndarray,
    robot_params: RobotParams,
    sim_params: SimParams,
    perturbation: np.ndarray | None = None,
    gaussian_noise_std_normalized: float | None = None,
    gaussian_noise_mask: np.ndarray | None = None,
    noise_seed: int | None = None,
) -> dict[str, np.ndarray]:
    state = initial_state.astype(np.float64).copy()
    states = []
    refs = []
    features = []
    wheels = []
    perturbations = []
    lookahead_steps = max(0, int(round(sim_params.lookahead_time / sim_params.dt)))
    rng = np.random.default_rng(noise_seed)
    if gaussian_noise_mask is None:
        noise_mask = np.ones_like(stats["x_std"], dtype=np.float32)
    else:
        noise_mask = _validate_policy_vector("gaussian_noise_mask", gaussian_noise_mask, int(stats["x_std"].shape[0]))
    if perturbation is not None:
        perturbation = _validate_policy_vector("perturbation", perturbation, int(stats["x_std"].shape[0]))

    for k in range(len(trajectory.t)):
        feature = tracking_features(state, trajectory, k, lookahead_steps)
        step_perturbation = perturbation
        if gaussian_noise_std_normalized is not None:
            noise_norm = rng.normal(0.0, gaussian_noise_std_normalized, size=feature.shape).astype(np.float32)
            step_perturbation = noise_norm * stats["x_std"] * noise_mask
        wr, wl = policy_wheels(model, stats, feature, robot_params, step_perturbation)
        states.append(state.copy())
        refs.append(np.array([trajectory.x[k], trajectory.y[k], trajectory.theta[k]], dtype=np.float64))
        features.append(feature)
        wheels.append(np.array([wr, wl], dtype=np.float64))
        perturbations.append(
            np.zeros_like(feature, dtype=np.float32)
            if step_perturbation is None
            else step_perturbation.astype(np.float32)
        )
        state = step_differential_drive(state, wr, wl, sim_params.dt, robot_params)

    return {
        "t": trajectory.t,
        "states": np.asarray(states),
        "refs": np.asarray(refs),
        "features": np.asarray(features),
        "wheels": np.asarray(wheels),
        "perturbations": np.asarray(perturbations),
    }


def compute_metrics(rollout: dict[str, np.ndarray]) -> dict[str, float]:
    err_xy = rollout["refs"][:, :2] - rollout["states"][:, :2]
    ep = np.linalg.norm(err_xy, axis=1)
    eth = np.abs(wrap_to_pi(rollout["refs"][:, 2] - rollout["states"][:, 2]))
    t = rollout["t"]
    return {
        "rmse_pos": float(np.sqrt(np.mean(ep**2))),
        "max_pos": float(np.max(ep)),
        "iae_pos": float(np.trapezoid(ep, t)),
        "rmse_theta": float(np.sqrt(np.mean(eth**2))),
        "max_theta": float(np.max(eth)),
    }
