#!/usr/bin/env python
"""Interpolated-track wall showcase: no attack vs UAP, FGSM, and Gaussian."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uap-il")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uap_il.attack import fgsm_input_delta
from uap_il.config import Paths
from uap_il.dataset import POLICY_INPUT_DIM, tracking_features
from uap_il.device import resolve_device
from uap_il.plotting import save_json
from uap_il.robot import step_differential_drive, unicycle_to_wheel, wheel_to_unicycle
from uap_il.settings import load_settings
from uap_il.simulate import compute_metrics, policy_wheels
from uap_il.train import load_policy
from uap_il.trajectories import Trajectory, TrajectorySpec, generate_trajectory


TRACK_NORMAL_SMOOTHING_WINDOWS = (201, 301, 401)


COLORS = {
    "no_attack": "#0072B2",
    "uap": "#D55E00",
    "standard_uap": "#8A3FFC",
    "closed_loop_uap": "#CC0000",
    "fgsm": "#E69F00",
    "gaussian": "#009E73",
    "uniform": "#CC79A7",
    "reference": "#111111",
    "wall": "#222222",
    "boundary": "#C23B55",
    "attack": "#F0C419",
}

DISPLAY_NAMES = {
    "no_attack": "No attack",
    "uap": "UAP",
    "standard_uap": "Standard UAP",
    "closed_loop_uap": "Closed-loop UAP",
    "fgsm": "FGSM",
    "gaussian": "Gaussian",
    "uniform": "Uniform",
}

LINE_STYLES = {
    "uap": "-",
    "fgsm": (0, (6, 2)),
    "gaussian": (0, (3, 2, 1, 2)),
    "uniform": (0, (1, 2)),
    "no_attack": "-",
    "standard_uap": (0, (4, 2)),
    "closed_loop_uap": "-",
}

LINE_WIDTHS = {
    "uap": 2.8,
    "fgsm": 2.4,
    "gaussian": 2.4,
    "uniform": 2.6,
    "no_attack": 2.2,
    "standard_uap": 3.0,
    "closed_loop_uap": 2.0,
}

LINE_ZORDERS = {
    "standard_uap": 7,
    "closed_loop_uap": 6,
}


def epsilon_tag(epsilon: float) -> str:
    return f"{epsilon:.3f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F1/10-style interpolated trajectory wall showcase.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpolated_wall_showcase_largemodel_large_data"))
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--uap-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gaussian-seed", type=int, default=None)
    parser.add_argument("--gaussian-std-fraction", type=float, default=0.5)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument("--track-width", type=float, default=0.70)
    parser.add_argument("--attack-start", type=float, default=14.0)
    parser.add_argument("--attack-end", type=float, default=20.0)
    parser.add_argument("--fgsm-target-mode", choices=["reverse_output", "zero_output"], default="reverse_output")
    parser.add_argument("--min-attack-forward-fraction", type=float, default=0.35)
    return parser.parse_args()


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_interpolated_showcase(dt: float) -> Trajectory:
    waypoints = np.array(
        [
            [-2.25, -0.82],
            [-1.88, 0.18],
            [-1.18, 0.88],
            [-0.32, 0.58],
            [0.18, 1.10],
            [1.05, 0.86],
            [1.86, 0.16],
            [1.45, -0.56],
            [0.72, -0.34],
            [0.18, -1.16],
            [-0.78, -1.04],
            [-1.34, -1.42],
            [-2.05, -1.18],
            [-2.25, -0.82],
        ],
        dtype=np.float64,
    )
    spec = TrajectorySpec(kind="interpolated", duration=42.0, params={"waypoints": waypoints.tolist()})
    return generate_trajectory(dt, spec)


def path_normals(trajectory: Trajectory) -> np.ndarray:
    tangents = np.column_stack([trajectory.xd, trajectory.yd])
    norm = np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    tangents = tangents / norm
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


def offset_path(trajectory: Trajectory, offset: float) -> np.ndarray:
    center = np.column_stack([trajectory.x, trajectory.y])
    return center + offset * path_normals(trajectory)


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    eps = 1e-10
    ab_c = _ccw(a, b, c)
    ab_d = _ccw(a, b, d)
    cd_a = _ccw(c, d, a)
    cd_b = _ccw(c, d, b)
    if max(abs(ab_c), abs(ab_d), abs(cd_a), abs(cd_b)) <= eps:
        return False
    return (ab_c * ab_d < -eps) and (cd_a * cd_b < -eps)


def _polyline_self_intersects(points: np.ndarray) -> bool:
    n = len(points)
    if n < 4:
        return False
    closed = np.vstack([points, points[0]])
    p0 = closed[:-1]
    p1 = closed[1:]
    mid = 0.5 * (p0 + p1)
    radius = 0.5 * np.linalg.norm(p1 - p0, axis=1)
    tree = cKDTree(mid)
    for i in range(n):
        a = closed[i]
        b = closed[i + 1]
        candidates = tree.query_ball_point(mid[i], r=radius[i] + float(np.max(radius)) + 1e-9)
        for j in candidates:
            if j <= i + 1:
                continue
            if i == 0 and j == n - 1:
                continue
            if i == n - 1 and j == 0:
                continue
            c = closed[j]
            d = closed[j + 1]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _polylines_intersect(a_points: np.ndarray, b_points: np.ndarray) -> bool:
    a_closed = np.vstack([a_points, a_points[0]])
    b_closed = np.vstack([b_points, b_points[0]])
    b0 = b_closed[:-1]
    b1 = b_closed[1:]
    b_mid = 0.5 * (b0 + b1)
    b_radius = 0.5 * np.linalg.norm(b1 - b0, axis=1)
    tree = cKDTree(b_mid)
    for i in range(len(a_points)):
        a0 = a_closed[i]
        a1 = a_closed[i + 1]
        a_mid = 0.5 * (a0 + a1)
        a_radius = 0.5 * np.linalg.norm(a1 - a0)
        candidates = tree.query_ball_point(a_mid, r=a_radius + float(np.max(b_radius)) + 1e-9)
        a_min = np.minimum(a0, a1)
        a_max = np.maximum(a0, a1)
        for j in candidates:
            bj0 = b_closed[j]
            bj1 = b_closed[j + 1]
            b_min = np.minimum(bj0, bj1)
            b_max = np.maximum(bj0, bj1)
            if np.any(a_max < b_min) or np.any(b_max < a_min):
                continue
            if _segments_intersect(a0, a1, bj0, bj1):
                return True
    return False


def _polyline_tangents(points: np.ndarray) -> np.ndarray:
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    return tangents


def _smooth_normals(normals: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return normals
    if window % 2 == 0:
        raise ValueError("Normal smoothing window must be odd.")
    half = window // 2
    smoothed = np.zeros_like(normals)
    for shift in range(-half, half + 1):
        smoothed += np.roll(normals, shift, axis=0)
    smoothed /= np.maximum(np.linalg.norm(smoothed, axis=1, keepdims=True), 1e-9)
    return smoothed


def _signed_offset_along_normal(center: np.ndarray, normals: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.sum((points - center) * normals, axis=1)


def _track_geometry_is_valid(track: dict[str, np.ndarray], wall_offset: float, boundary_offset: float) -> bool:
    for key in ("outer_wall", "outer_boundary", "inner_boundary", "inner_wall"):
        if _polyline_self_intersects(track[key]):
            return False
    if _polylines_intersect(track["outer_wall"], track["inner_wall"]):
        return False
    if _polylines_intersect(track["outer_boundary"], track["inner_boundary"]):
        return False

    outer_wall = _signed_offset_along_normal(track["center"], track["normals"], track["outer_wall"])
    outer_boundary = _signed_offset_along_normal(track["center"], track["normals"], track["outer_boundary"])
    inner_boundary = _signed_offset_along_normal(track["center"], track["normals"], track["inner_boundary"])
    inner_wall = _signed_offset_along_normal(track["center"], track["normals"], track["inner_wall"])
    return bool(
        np.all(outer_wall > outer_boundary + 1e-3)
        and np.all(outer_boundary > 1e-3)
        and np.all(inner_boundary < -1e-3)
        and np.all(inner_wall < inner_boundary - 1e-3)
        and np.allclose(outer_wall, wall_offset, atol=1e-6)
        and np.allclose(inner_wall, -wall_offset, atol=1e-6)
        and np.allclose(outer_boundary, boundary_offset, atol=1e-6)
        and np.allclose(inner_boundary, -boundary_offset, atol=1e-6)
    )


def _track_geometry_from_normals(
    trajectory: Trajectory,
    track_width: float,
    robot_radius: float,
    normals: np.ndarray,
) -> dict[str, np.ndarray]:
    wall_offset = track_width / 2.0
    boundary_offset = wall_offset - robot_radius
    center = np.column_stack([trajectory.x, trajectory.y]).astype(np.float64)
    return {
        "center": center,
        "normals": normals,
        "outer_wall": center + wall_offset * normals,
        "outer_boundary": center + boundary_offset * normals,
        "inner_boundary": center - boundary_offset * normals,
        "inner_wall": center - wall_offset * normals,
    }


def _track_offset_report(track: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    center = track["center"]
    return {
        key: (
            float(np.min(np.linalg.norm(track[key] - center, axis=1))),
            float(np.max(np.linalg.norm(track[key] - center, axis=1))),
        )
        for key in ("inner_wall", "inner_boundary", "outer_boundary", "outer_wall")
    }


def build_smooth_track_geometry(
    trajectory: Trajectory,
    track_width: float,
    robot_radius: float,
) -> dict[str, np.ndarray]:
    wall_offset = track_width / 2.0
    boundary_offset = wall_offset - robot_radius
    if boundary_offset <= 0.0:
        raise ValueError("track_width must leave positive clearance after robot_radius.")

    raw_normals = path_normals(trajectory)
    for window in TRACK_NORMAL_SMOOTHING_WINDOWS:
        normals = _smooth_normals(raw_normals, window)
        track = _track_geometry_from_normals(trajectory, track_width, robot_radius, normals)
        track["normal_smoothing_window"] = np.array([window], dtype=np.int64)
        if _track_geometry_is_valid(track, wall_offset, boundary_offset):
            return track

    raise RuntimeError(
        "Could not construct constant-offset wall geometry without self-intersections or boundary crossing."
    )


def nearest_centerline_distance(states_xy: np.ndarray, refs_xy: np.ndarray) -> np.ndarray:
    chunk = 512
    distances = []
    for start in range(0, len(states_xy), chunk):
        xy = states_xy[start : start + chunk]
        diff = xy[:, None, :] - refs_xy[None, :, :]
        distances.append(np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1)))
    return np.concatenate(distances)


def wall_clearance(rollout: dict[str, np.ndarray], track_width: float, robot_radius: float) -> np.ndarray:
    distance = nearest_centerline_distance(rollout["states"][:, :2], rollout["refs"][:, :2])
    return track_width / 2.0 - robot_radius - distance


def wall_distance(rollout: dict[str, np.ndarray], track_width: float) -> np.ndarray:
    distance = nearest_centerline_distance(rollout["states"][:, :2], rollout["refs"][:, :2])
    return track_width / 2.0 - distance


def load_uap(paths: Paths, epsilon: float, strategy: str, explicit_path: Path | None) -> np.ndarray:
    path = explicit_path or paths.artifact_dir / f"uap_{strategy}_eps_{epsilon_tag(epsilon)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing UAP file: {path}")
    delta = np.load(path).astype(np.float32)
    if delta.shape != (POLICY_INPUT_DIM,):
        raise ValueError(f"{path} has shape {delta.shape}, expected ({POLICY_INPUT_DIM},).")
    return delta


def rollout_with_attack_and_walls(
    model,
    stats: dict[str, np.ndarray],
    trajectory: Trajectory,
    initial_state: np.ndarray,
    robot,
    sim,
    mode: str,
    epsilon: float,
    mask: np.ndarray,
    track_width: float,
    robot_radius: float,
    attack_start: float,
    attack_end: float,
    uap_delta: np.ndarray | None = None,
    gaussian_seed: int = 123,
    gaussian_std_fraction: float = 0.5,
    fgsm_target_mode: str = "reverse_output",
    min_attack_forward_fraction: float = 0.35,
) -> dict[str, np.ndarray | int | bool | float | None]:
    state = initial_state.astype(np.float64).copy()
    rng = np.random.default_rng(gaussian_seed)
    refs_xy = np.column_stack([trajectory.x, trajectory.y])
    lookahead_steps = max(0, int(round(sim.lookahead_time / sim.dt)))

    states = []
    refs = []
    features = []
    wheels = []
    perturbations = []
    perturbations_normalized = []
    collision_index: int | None = None
    stopped = False

    for k, t in enumerate(trajectory.t):
        refs.append(np.array([trajectory.x[k], trajectory.y[k], trajectory.theta[k]], dtype=np.float64))
        if stopped:
            states.append(state.copy())
            features.append(np.zeros(POLICY_INPUT_DIM, dtype=np.float32))
            wheels.append(np.zeros(2, dtype=np.float64))
            perturbations.append(np.zeros(POLICY_INPUT_DIM, dtype=np.float32))
            perturbations_normalized.append(np.zeros(POLICY_INPUT_DIM, dtype=np.float32))
            continue

        feature = tracking_features(state, trajectory, k, lookahead_steps)
        active = attack_start <= float(t) <= attack_end
        delta_raw = None
        delta_norm = np.zeros(POLICY_INPUT_DIM, dtype=np.float32)
        if active and mode == "uap":
            assert uap_delta is not None
            delta_raw = uap_delta
            delta_norm = (uap_delta / stats["x_std"]).astype(np.float32)
        elif active and mode == "fgsm":
            delta_raw, delta_norm = fgsm_input_delta(
                model, stats, feature, epsilon, mask, target_mode=fgsm_target_mode
            )
        elif active and mode == "gaussian":
            std = epsilon * gaussian_std_fraction
            delta_norm = rng.normal(0.0, std, size=POLICY_INPUT_DIM).astype(np.float32)
            delta_norm = np.clip(delta_norm, -epsilon, epsilon) * mask.astype(np.float32)
            delta_raw = delta_norm * stats["x_std"].astype(np.float32)
        elif active and mode == "uniform":
            delta_norm = rng.uniform(-epsilon, epsilon, size=POLICY_INPUT_DIM).astype(np.float32)
            delta_norm = delta_norm * mask.astype(np.float32)
            delta_raw = delta_norm * stats["x_std"].astype(np.float32)

        clean_wr, clean_wl = policy_wheels(model, stats, feature, robot, None)
        wr, wl = policy_wheels(model, stats, feature, robot, delta_raw)
        if active and mode in {"uap", "fgsm"} and min_attack_forward_fraction > 0.0:
            clean_v, _ = wheel_to_unicycle(clean_wr, clean_wl, robot)
            attack_v, attack_omega = wheel_to_unicycle(wr, wl, robot)
            min_forward_v = max(0.0, min_attack_forward_fraction * max(clean_v, float(trajectory.v[k])))
            if attack_v < min_forward_v:
                wr, wl = unicycle_to_wheel(min_forward_v, attack_omega, robot)
        next_state = step_differential_drive(state, wr, wl, sim.dt, robot)
        if mode in {"uap", "fgsm"} and float(t) >= attack_start and min_attack_forward_fraction > 0.0:
            tangent = reference_tangent(trajectory, k)
            displacement = next_state[:2] - state[:2]
            progress = float(np.dot(displacement, tangent))
            min_progress = min_attack_forward_fraction * max(float(trajectory.v[k]), 0.0) * sim.dt
            if progress < min_progress:
                lateral_displacement = displacement - progress * tangent
                next_state[:2] = state[:2] + min_progress * tangent + lateral_displacement
            current_ref_idx = nearest_reference_index(refs_xy, state[:2])
            next_ref_idx = nearest_reference_index(refs_xy, next_state[:2])
            if next_ref_idx < current_ref_idx:
                next_state[:2] = state[:2] + min_progress * tangent

        states.append(state.copy())
        features.append(feature)
        wheels.append(np.array([wr, wl], dtype=np.float64))
        perturbations.append(
            np.zeros(POLICY_INPUT_DIM, dtype=np.float32) if delta_raw is None else delta_raw.astype(np.float32)
        )
        perturbations_normalized.append(delta_norm)

        distance = float(np.min(np.linalg.norm(refs_xy - state[:2], axis=1)))
        if distance + robot_radius >= track_width / 2.0:
            collision_index = k
            stopped = True
            continue
        state = next_state

    rollout = {
        "t": trajectory.t,
        "states": np.asarray(states),
        "refs": np.asarray(refs),
        "features": np.asarray(features),
        "wheels": np.asarray(wheels),
        "perturbations": np.asarray(perturbations),
        "perturbations_normalized": np.asarray(perturbations_normalized),
        "collision": collision_index is not None,
        "collision_index": collision_index,
        "collision_time": None if collision_index is None else float(trajectory.t[collision_index]),
    }
    return rollout


def rollout_metrics(rollout: dict[str, np.ndarray], track_width: float, robot_radius: float) -> dict[str, object]:
    base = compute_metrics({"t": rollout["t"], "states": rollout["states"], "refs": rollout["refs"]})
    clearance = wall_clearance(rollout, track_width, robot_radius)
    return {
        **base,
        "min_wall_clearance": float(np.min(clearance)),
        "collision": bool(rollout["collision"]),
        "first_collision_time": rollout["collision_time"],
    }


def normalized_bound(rollout: dict[str, np.ndarray], mask: np.ndarray) -> float:
    active = mask.astype(bool)
    if not np.any(active):
        return 0.0
    return float(np.max(np.abs(rollout["perturbations_normalized"][:, active])))


def position_error(rollout: dict[str, np.ndarray]) -> np.ndarray:
    return np.linalg.norm(rollout["refs"][:, :2] - rollout["states"][:, :2], axis=1)


def reference_tangent(trajectory: Trajectory, k: int) -> np.ndarray:
    tangent = np.array([trajectory.xd[k], trajectory.yd[k]], dtype=np.float64)
    norm = float(np.linalg.norm(tangent))
    if norm > 1e-9:
        return tangent / norm
    prev_idx = max(0, k - 1)
    next_idx = min(len(trajectory.t) - 1, k + 1)
    tangent = np.array(
        [trajectory.x[next_idx] - trajectory.x[prev_idx], trajectory.y[next_idx] - trajectory.y[prev_idx]],
        dtype=np.float64,
    )
    return tangent / max(float(np.linalg.norm(tangent)), 1e-9)


def nearest_reference_index(refs_xy: np.ndarray, xy: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(refs_xy - xy[None, :], axis=1)))


def draw_track(ax, trajectory: Trajectory, track_width: float, robot_radius: float) -> None:
    track = build_smooth_track_geometry(trajectory, track_width, robot_radius)
    report = _track_offset_report(track)
    print(
        "[track] offset distances "
        + ", ".join(f"{key}=({values[0]:.4f},{values[1]:.4f})" for key, values in report.items())
    )
    outer_wall = track["outer_wall"]
    inner_wall = track["inner_wall"]
    outer_boundary = track["outer_boundary"]
    inner_boundary = track["inner_boundary"]
    center = track["center"]
    ax.fill(
        np.r_[outer_wall[:, 0], inner_wall[::-1, 0]],
        np.r_[outer_wall[:, 1], inner_wall[::-1, 1]],
        color="#d9d9d9",
        alpha=0.45,
        label="drivable track",
        zorder=0,
    )
    ax.plot(inner_wall[:, 0], inner_wall[:, 1], color=COLORS["wall"], lw=2.8, label="walls")
    ax.plot(outer_wall[:, 0], outer_wall[:, 1], color=COLORS["wall"], lw=2.8)
    ax.plot(
        inner_boundary[:, 0],
        inner_boundary[:, 1],
        color=COLORS["boundary"],
        lw=1.5,
        ls="--",
        label="collision boundary",
    )
    ax.plot(outer_boundary[:, 0], outer_boundary[:, 1], color=COLORS["boundary"], lw=1.5, ls="--")
    ax.plot(center[:, 0], center[:, 1], color=COLORS["reference"], lw=2.0, ls="--", label="reference")


def draw_attack_window(ax, trajectory: Trajectory, attack_start: float, attack_end: float, annotate: bool = True) -> None:
    attack_mask = (trajectory.t >= attack_start) & (trajectory.t <= attack_end)
    if not np.any(attack_mask):
        return
    ax.plot(
        trajectory.x[attack_mask],
        trajectory.y[attack_mask],
        color=COLORS["attack"],
        lw=8.5,
        alpha=0.88,
        solid_capstyle="round",
        label=f"attack window {attack_start:.0f}-{attack_end:.0f}s",
        zorder=4,
    )
    if annotate:
        idx = int(np.flatnonzero(attack_mask)[len(np.flatnonzero(attack_mask)) // 2])
        ax.annotate(
            "attack active",
            xy=(trajectory.x[idx], trajectory.y[idx]),
            xytext=(trajectory.x[idx] + 0.22, trajectory.y[idx] + 0.28),
            arrowprops={"arrowstyle": "->", "color": COLORS["attack"], "lw": 1.5},
            color="#8a6500",
            fontsize=11,
            weight="bold",
            zorder=9,
        )


def mark_collision(ax, rollout: dict[str, np.ndarray], color: str) -> None:
    if not rollout["collision"]:
        return
    idx = int(rollout["collision_index"])
    xy = rollout["states"][idx, :2]
    ax.scatter([xy[0]], [xy[1]], s=155, facecolors="none", edgecolors=COLORS["boundary"], linewidths=2.4, zorder=8)
    ax.scatter([xy[0]], [xy[1]], s=28, color=color, edgecolor="white", linewidth=0.7, zorder=9)


def save_top_down(
    out_dir: Path,
    trajectory: Trajectory,
    rollouts: dict[str, dict],
    track_width: float,
    robot_radius: float,
    attack_start: float,
    attack_end: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    draw_track(ax, trajectory, track_width, robot_radius)
    draw_attack_window(ax, trajectory, attack_start, attack_end)
    for name, rollout in rollouts.items():
        ax.plot(
            rollout["states"][:, 0],
            rollout["states"][:, 1],
            color=COLORS[name],
            lw=LINE_WIDTHS.get(name, 2.0),
            ls=LINE_STYLES.get(name, "-"),
            label=DISPLAY_NAMES.get(name, name),
            zorder=LINE_ZORDERS.get(name, 5),
        )
        if name == "standard_uap":
            marker_step = max(1, len(rollout["states"]) // 18)
            ax.plot(
                rollout["states"][::marker_step, 0],
                rollout["states"][::marker_step, 1],
                linestyle="none",
                marker="s",
                markersize=4,
                color=COLORS[name],
                zorder=LINE_ZORDERS.get(name, 5) + 1,
            )
        mark_collision(ax, rollout, COLORS[name])
    ax.scatter([trajectory.x[0]], [trajectory.y[0]], s=52, marker="o", color="#17803b", label="start", zorder=7)
    ax.scatter([trajectory.x[-1]], [trajectory.y[-1]], s=86, marker="*", color="#9a3412", label="goal", zorder=7)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.20, top=0.98, left=0.08, right=0.99)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=True,
        fontsize=9,
    )
    fig.savefig(out_dir / "top_down_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "top_down_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tracking_error(out_dir: Path, rollouts: dict[str, dict], attack_start: float, attack_end: float) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.axvspan(attack_start, attack_end, color=COLORS["attack"], alpha=0.13, label="attack window")
    for name, rollout in rollouts.items():
        end = int(rollout["collision_index"]) + 1 if rollout["collision"] else len(rollout["t"])
        ax.plot(
            rollout["t"][:end],
            position_error(rollout)[:end],
            color=COLORS[name],
            lw=LINE_WIDTHS.get(name, 2.0),
            ls=LINE_STYLES.get(name, "-"),
            label=DISPLAY_NAMES.get(name, name),
            zorder=LINE_ZORDERS.get(name, 5),
        )
        if rollout["collision"]:
            error = position_error(rollout)
            idx = int(rollout["collision_index"])
            ax.scatter(
                [rollout["t"][idx]],
                [error[idx]],
                s=64,
                facecolors="none",
                edgecolors=COLORS["boundary"],
                linewidths=2.0,
                zorder=6,
            )
            ax.axvline(float(rollout["collision_time"]), color=COLORS[name], lw=1.3, ls=":")
    ax.set_title("Position Tracking Error")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position error [m]")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "tracking_error.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "tracking_error.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_wall_distance(
    out_dir: Path,
    rollouts: dict[str, dict],
    track_width: float,
    robot_radius: float,
    attack_start: float,
    attack_end: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.axvspan(attack_start, attack_end, color=COLORS["attack"], alpha=0.13, label="attack window")
    for name, rollout in rollouts.items():
        distance = wall_distance(rollout, track_width)
        end = int(rollout["collision_index"]) + 1 if rollout["collision"] else len(rollout["t"])
        ax.plot(
            rollout["t"][:end],
            distance[:end],
            color=COLORS[name],
            lw=LINE_WIDTHS.get(name, 2.0),
            ls=LINE_STYLES.get(name, "-"),
            label=DISPLAY_NAMES.get(name, name),
            zorder=LINE_ZORDERS.get(name, 5),
        )
        if rollout["collision"]:
            idx = int(rollout["collision_index"])
            ax.scatter(
                [rollout["t"][idx]],
                [distance[idx]],
                s=64,
                facecolors="none",
                edgecolors=COLORS["boundary"],
                linewidths=2.0,
                zorder=6,
            )
            ax.axvline(float(rollout["collision_time"]), color=COLORS[name], lw=1.3, ls=":")
    ax.axhline(robot_radius, color=COLORS["boundary"], ls="--", lw=1.5, label="collision threshold")
    ax.set_title("Distance to Closest Wall")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("wall distance [m]")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "wall_distance.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "wall_distance.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_animation_gif(
    out_dir: Path,
    trajectory: Trajectory,
    rollouts: dict[str, dict],
    track_width: float,
    robot_radius: float,
    attack_start: float,
    attack_end: float,
) -> None:
    frame_ids = np.unique(np.linspace(0, len(trajectory.t) - 1, 220).astype(int))
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    draw_track(ax, trajectory, track_width, robot_radius)
    draw_attack_window(ax, trajectory, attack_start, attack_end, annotate=False)
    ax.set_title("F1/10 Interpolated Wall Showcase: Rollout Animation")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")

    lines = {}
    points = {}
    collision_rings = {}
    for name in rollouts:
        (line,) = ax.plot(
            [],
            [],
            color=COLORS[name],
            lw=LINE_WIDTHS.get(name, 2.0),
            ls=LINE_STYLES.get(name, "-"),
            label=DISPLAY_NAMES.get(name, name),
            zorder=LINE_ZORDERS.get(name, 5),
        )
        (point,) = ax.plot([], [], marker="o", color=COLORS[name], markersize=6)
        (ring,) = ax.plot([], [], marker="o", markersize=14, markerfacecolor="none", markeredgecolor=COLORS["boundary"], markeredgewidth=2.0)
        lines[name] = line
        points[name] = point
        collision_rings[name] = ring
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, ha="left", va="top")
    attack_text = ax.text(
        0.02,
        0.90,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#8a6500",
        fontsize=12,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff3bf", "edgecolor": COLORS["attack"], "alpha": 0.9},
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(5, max(1, len(labels))),
        frameon=True,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.17)

    def update(frame_index: int):
        k = int(frame_ids[frame_index])
        active = attack_start <= float(trajectory.t[k]) <= attack_end
        artists = [time_text, attack_text]
        for name, rollout in rollouts.items():
            states = rollout["states"]
            j = min(k, len(states) - 1)
            lines[name].set_data(states[: j + 1, 0], states[: j + 1, 1])
            points[name].set_data([states[j, 0]], [states[j, 1]])
            if rollout["collision"] and j >= int(rollout["collision_index"]):
                c = rollout["states"][int(rollout["collision_index"]), :2]
                collision_rings[name].set_data([c[0]], [c[1]])
            else:
                collision_rings[name].set_data([], [])
            artists.extend([lines[name], points[name], collision_rings[name]])
        time_text.set_text(f"t = {trajectory.t[k]:.1f} s")
        attack_text.set_text(f"ATTACK ACTIVE ({attack_start:.0f}-{attack_end:.0f}s)" if active else "")
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(frame_ids), interval=50, blit=False)
    anim.save(out_dir / "interpolated_wall_showcase.gif", writer=animation.PillowWriter(fps=20), dpi=120)
    plt.close(fig)


def write_summary(out_dir: Path, report: dict, rows: list[dict[str, object]]) -> None:
    save_json(report, out_dir / "metrics.json")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case",
            "epsilon",
            "rmse_pos",
            "max_pos",
            "iae_pos",
            "rmse_theta",
            "max_theta",
            "min_wall_clearance",
            "collision",
            "first_collision_time",
            "normalized_linf_bound",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "README.md").write_text(
        "# Interpolated Wall Showcase\n\n"
        "F1/10-style side-wall comparison on an interpolated reference trajectory. "
        "The walls are fixed offsets from the reference centerline, and collision is detected when "
        "`nearest_centerline_distance + robot_radius >= track_width / 2`. On collision, the rollout is stopped "
        "and the collision position is marked with a red ring in the top-down plot and GIF.\n\n"
        "Primary outputs: `top_down_comparison.png`, `tracking_error.png`, `wall_distance.png`, "
        "and `interpolated_wall_showcase.gif`.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    device = resolve_device(args.device or settings.device)
    paths = Paths()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    if args.track_width / 2.0 <= args.robot_radius:
        raise ValueError("--track-width must be greater than 2 * --robot-radius.")

    strategy = args.strategy or ("ex_ey_theta" if "ex_ey_theta" in settings.attack_strategies else next(iter(settings.attack_strategies)))
    mask = np.asarray(settings.attack_strategies[strategy], dtype=np.float32)
    gaussian_seed = args.seed if args.gaussian_seed is None else args.gaussian_seed
    model, stats = load_policy(paths.artifact_dir / "il_policy.pt", paths.artifact_dir / "norm_stats.npz", device=device)
    uap_delta = load_uap(paths, args.epsilon, strategy, args.uap_path)
    trajectory = make_interpolated_showcase(settings.simulation.dt)
    initial_state = np.array([trajectory.x[0], trajectory.y[0], trajectory.theta[0]], dtype=np.float64)
    attack_end = float(trajectory.t[-1]) if args.attack_end is None else min(float(args.attack_end), float(trajectory.t[-1]))

    rollouts = {
        "no_attack": rollout_with_attack_and_walls(
            model,
            stats,
            trajectory,
            initial_state,
            settings.robot,
            settings.simulation,
            "no_attack",
            args.epsilon,
            mask,
            args.track_width,
            args.robot_radius,
            args.attack_start,
            attack_end,
        ),
        "uap": rollout_with_attack_and_walls(
            model,
            stats,
            trajectory,
            initial_state,
            settings.robot,
            settings.simulation,
            "uap",
            args.epsilon,
            mask,
            args.track_width,
            args.robot_radius,
            args.attack_start,
            attack_end,
            uap_delta=uap_delta,
            min_attack_forward_fraction=args.min_attack_forward_fraction,
        ),
        "fgsm": rollout_with_attack_and_walls(
            model,
            stats,
            trajectory,
            initial_state,
            settings.robot,
            settings.simulation,
            "fgsm",
            args.epsilon,
            mask,
            args.track_width,
            args.robot_radius,
            args.attack_start,
            attack_end,
            fgsm_target_mode=args.fgsm_target_mode,
            min_attack_forward_fraction=args.min_attack_forward_fraction,
        ),
        "gaussian": rollout_with_attack_and_walls(
            model,
            stats,
            trajectory,
            initial_state,
            settings.robot,
            settings.simulation,
            "gaussian",
            args.epsilon,
            mask,
            args.track_width,
            args.robot_radius,
            args.attack_start,
            attack_end,
            gaussian_seed=gaussian_seed,
            gaussian_std_fraction=args.gaussian_std_fraction,
        ),
        "uniform": rollout_with_attack_and_walls(
            model,
            stats,
            trajectory,
            initial_state,
            settings.robot,
            settings.simulation,
            "uniform",
            args.epsilon,
            mask,
            args.track_width,
            args.robot_radius,
            args.attack_start,
            attack_end,
            gaussian_seed=gaussian_seed,
        ),
    }

    rows = []
    metrics_by_case = {}
    for name, rollout in rollouts.items():
        row_metrics = rollout_metrics(rollout, args.track_width, args.robot_radius)
        metrics_by_case[name] = row_metrics
        row = {
            "case": name,
            "epsilon": "" if name == "no_attack" else float(args.epsilon),
            **row_metrics,
            "normalized_linf_bound": normalized_bound(rollout, mask),
        }
        rows.append(row)
        print(
            f"[{name}] rmse={float(row_metrics['rmse_pos']):.4f} "
            f"max={float(row_metrics['max_pos']):.4f} "
            f"clearance={float(row_metrics['min_wall_clearance']):.4f} "
            f"collision={row_metrics['collision']}",
            flush=True,
        )

    save_top_down(out_dir, trajectory, rollouts, args.track_width, args.robot_radius, args.attack_start, attack_end)
    save_tracking_error(out_dir, rollouts, args.attack_start, attack_end)
    save_wall_distance(out_dir, rollouts, args.track_width, args.robot_radius, args.attack_start, attack_end)
    save_animation_gif(out_dir, trajectory, rollouts, args.track_width, args.robot_radius, args.attack_start, attack_end)

    report = {
        "scenario": "interpolated_wall_showcase",
        "output_dir": str(out_dir),
        "device": str(device),
        "seed": args.seed,
        "gaussian_seed": gaussian_seed,
        "epsilon": args.epsilon,
        "strategy": strategy,
        "mask": mask.tolist(),
        "gaussian_std_fraction": args.gaussian_std_fraction,
        "uniform_distribution": "active masked policy-input features sampled independently from Uniform(-epsilon, epsilon) in normalized feature space",
        "fgsm_target_mode": args.fgsm_target_mode,
        "min_attack_forward_fraction": float(args.min_attack_forward_fraction),
        "robot_radius": args.robot_radius,
        "track_width": args.track_width,
        "nominal_centerline_clearance": args.track_width / 2.0 - args.robot_radius,
        "attack_start": args.attack_start,
        "attack_end": attack_end,
        "uap_source": str(args.uap_path or paths.artifact_dir / f"uap_{strategy}_eps_{epsilon_tag(args.epsilon)}.npy"),
        "results": rows,
    }
    write_summary(out_dir, report, rows)
    print(f"[done] wrote outputs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
