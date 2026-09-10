#!/usr/bin/env python
"""Stress-test old and retrained policies on the interpolated wall under randomized dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uap-il")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import interpolated_wall_showcase as wall
from uap_il.config import RobotParams
from uap_il.dataset import tracking_features
from uap_il.device import resolve_device
from uap_il.robot import step_differential_drive, wheel_to_unicycle
from uap_il.settings import load_settings
from uap_il.simulate import compute_metrics, policy_wheels
from uap_il.train import load_policy


@dataclass(frozen=True)
class Domain:
    wheel_radius_scale: float
    wheelbase_scale: float
    right_wheel_gain: float
    left_wheel_gain: float
    motor_time_constant_s: float
    action_delay_steps: int
    observation_delay_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--include-legacy", action="store_true", help="Also evaluate older baseline checkpoints when available.")
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sim2real_interpolated_wall_stress"))
    return parser.parse_args()


def sample_domains(count: int, seed: int) -> list[Domain]:
    rng = np.random.default_rng(seed)
    return [
        Domain(
            wheel_radius_scale=float(rng.uniform(0.94, 1.06)),
            wheelbase_scale=float(rng.uniform(0.94, 1.06)),
            right_wheel_gain=float(rng.uniform(0.88, 1.12)),
            left_wheel_gain=float(rng.uniform(0.88, 1.12)),
            motor_time_constant_s=float(rng.uniform(0.02, 0.12)),
            action_delay_steps=int(rng.integers(0, 4)),
            observation_delay_steps=int(rng.integers(0, 3)),
        )
        for _ in range(count)
    ]


def rollout(model, stats, trajectory, robot, sim, domain: Domain) -> dict[str, np.ndarray]:
    state = np.array([trajectory.x[0], trajectory.y[0], trajectory.theta[0]], dtype=np.float64)
    physical_robot = RobotParams(
        wheel_radius=robot.wheel_radius * domain.wheel_radius_scale,
        wheelbase=robot.wheelbase * domain.wheelbase_scale,
        wheel_speed_limit=robot.wheel_speed_limit,
    )
    state_history = [state.copy()]
    command_queue = [(0.0, 0.0)] * domain.action_delay_steps
    applied_wr = 0.0
    applied_wl = 0.0
    states, refs, features, wheels = [], [], [], []
    lookahead_steps = max(0, int(round(sim.lookahead_time / sim.dt)))
    for k in range(len(trajectory.t)):
        observed = state_history[max(0, len(state_history) - 1 - domain.observation_delay_steps)]
        feature = tracking_features(observed, trajectory, k, lookahead_steps)
        wr, wl = policy_wheels(model, stats, feature, robot)
        command_queue.append((wr, wl))
        delayed_wr, delayed_wl = command_queue.pop(0)
        alpha = 1.0 - np.exp(-sim.dt / domain.motor_time_constant_s)
        applied_wr += alpha * (delayed_wr - applied_wr)
        applied_wl += alpha * (delayed_wl - applied_wl)
        physical_wr = domain.right_wheel_gain * applied_wr
        physical_wl = domain.left_wheel_gain * applied_wl
        states.append(state.copy())
        refs.append(np.array([trajectory.x[k], trajectory.y[k], trajectory.theta[k]]))
        features.append(feature)
        wheels.append(np.array([wr, wl]))
        state = step_differential_drive(state, physical_wr, physical_wl, sim.dt, physical_robot)
        state_history.append(state.copy())
    return {
        "t": trajectory.t,
        "states": np.asarray(states),
        "refs": np.asarray(refs),
        "features": np.asarray(features),
        "wheels": np.asarray(wheels),
    }


def rollout_metrics(result: dict[str, np.ndarray], track_width: float, robot_radius: float) -> dict[str, float | bool]:
    metrics = compute_metrics(result)
    clearance = wall.wall_clearance(result, track_width, robot_radius)
    steering = result["wheels"][:, 0] - result["wheels"][:, 1]
    steering_rate = np.diff(steering) / np.diff(result["t"])
    return {
        **metrics,
        "min_wall_clearance": float(np.min(clearance)),
        "collision": bool(np.min(clearance) <= 0.0),
        "steering_rate_rms": float(np.sqrt(np.mean(steering_rate**2))),
        "steering_total_variation": float(np.sum(np.abs(np.diff(steering)))),
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    policies = sorted({str(row["policy"]) for row in rows})
    metric_names = ["rmse_pos", "max_pos", "iae_pos", "rmse_theta", "min_wall_clearance", "steering_rate_rms", "steering_total_variation"]
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        entry: dict[str, object] = {"policy": policy, "trials": len(selected)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected])
            entry[f"{metric}_mean"] = float(np.mean(values))
            entry[f"{metric}_std"] = float(np.std(values))
            entry[f"{metric}_p95"] = float(np.percentile(values, 95))
        entry["collision_rate"] = float(np.mean([bool(row["collision"]) for row in selected]))
        result.append(entry)
    return result


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory = wall.make_interpolated_showcase(settings.simulation.dt)
    domains = sample_domains(args.trials, args.seed)
    policy_files = {
        "new_nominal": (Path("artifacts/sim2real_nominal_policy.pt"), Path("artifacts/sim2real_nominal_norm_stats.npz")),
        "new_adversarial": (Path("artifacts/sim2real_adversarial_policy.pt"), Path("artifacts/sim2real_adversarial_norm_stats.npz")),
    }
    if args.include_legacy:
        policy_files.update({
            "old_nominal": (Path("artifacts/il_policy.pt"), Path("artifacts/norm_stats.npz")),
            "old_adversarial": (Path("artifacts/residual_fgsm_multiscale_policy.pt"), Path("artifacts/residual_fgsm_multiscale_norm_stats.npz")),
        })
    rows: list[dict[str, object]] = []
    representative: dict[str, dict[str, np.ndarray]] = {}
    for policy_name, (checkpoint, norm) in policy_files.items():
        model, stats = load_policy(checkpoint, norm, device=device)
        for trial, domain in enumerate(domains):
            result = rollout(model, stats, trajectory, settings.robot, settings.simulation, domain)
            metrics = rollout_metrics(result, 0.70, 0.18)
            rows.append({"policy": policy_name, "trial": trial, **asdict(domain), **metrics})
            if trial == 0:
                representative[policy_name] = result
                np.savez_compressed(trajectory_dir / f"{policy_name}_trial_000.npz", **result)
        print(f"[eval] {policy_name}: {args.trials} randomized trials", flush=True)

    with (args.output_dir / "per_trial.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = aggregate(rows)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"domains": [asdict(domain) for domain in domains], "summary": summary}, indent=2), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.plot(trajectory.x, trajectory.y, "k--", lw=2.0, label="reference")
    for name, result in representative.items():
        ax.plot(result["states"][:, 0], result["states"][:, 1], lw=1.7, label=name.replace("_", " "))
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Interpolated-Wall Randomized-Dynamics Trial 0")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "trajectory_comparison.png", dpi=220)
    plt.close(fig)

    labels = [str(row["policy"]).replace("_", "\n") for row in summary]
    means = [float(row["rmse_pos_mean"]) for row in summary]
    stds = [float(row["rmse_pos_std"]) for row in summary]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(labels, means, yerr=stds, capsize=4)
    ax.set(ylabel="position RMSE [m]", title="Randomized-Dynamics Interpolated-Wall Tracking")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "tracking_rmse_comparison.png", dpi=220)
    plt.close(fig)
    print(f"[done] outputs -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
