from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import RobotParams
from .robot import unicycle_to_wheel


@dataclass(frozen=True)
class TrajectorySpec:
    kind: str
    duration: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xd: np.ndarray
    yd: np.ndarray
    xdd: np.ndarray
    ydd: np.ndarray
    theta: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    spec: TrajectorySpec


def _trajectory_from_xy(t: np.ndarray, x: np.ndarray, y: np.ndarray, spec: TrajectorySpec) -> Trajectory:
    xd = np.gradient(x, t, edge_order=2)
    yd = np.gradient(y, t, edge_order=2)
    xdd = np.gradient(xd, t, edge_order=2)
    ydd = np.gradient(yd, t, edge_order=2)
    theta = np.unwrap(np.arctan2(yd, xd))
    v = np.sqrt(xd**2 + yd**2)
    denom = np.maximum(xd**2 + yd**2, 1e-9)
    omega = (ydd * xd - xdd * yd) / denom
    return Trajectory(t, x, y, xd, yd, xdd, ydd, theta, v, omega, spec)


def _reference_wheel_peak(traj: Trajectory, robot_params: RobotParams) -> float:
    wr, wl = unicycle_to_wheel(traj.v, traj.omega, robot_params)
    return float(max(np.max(np.abs(wr)), np.max(np.abs(wl))))


def _reference_wheel_mean_abs(traj: Trajectory, robot_params: RobotParams) -> float:
    wr, wl = unicycle_to_wheel(traj.v, traj.omega, robot_params)
    return float(np.mean(np.abs(np.column_stack([wr, wl]))))


def reference_is_feasible(traj: Trajectory, robot_params: RobotParams, margin: float = 0.95) -> bool:
    peak = _reference_wheel_peak(traj, robot_params)
    return bool(np.isfinite(peak) and peak <= margin * robot_params.wheel_speed_limit)


def _scale_spec_time(spec: TrajectorySpec, factor: float) -> TrajectorySpec:
    factor = float(np.clip(factor, 0.08, 5.0))
    params = dict(spec.params)
    if spec.kind == "lemniscate":
        params["alpha"] = max(0.15, float(params["alpha"]) * factor)
        return TrajectorySpec(kind=spec.kind, duration=0.0, params=params)
    return TrajectorySpec(kind=spec.kind, duration=max(1.0, spec.duration * factor), params=params)


def _tune_reference_speed(
    spec: TrajectorySpec,
    dt: float,
    robot_params: RobotParams,
    target_mean_abs_wheel: float,
) -> TrajectorySpec:
    traj = generate_trajectory(dt, spec)
    mean_abs = _reference_wheel_mean_abs(traj, robot_params)
    peak = _reference_wheel_peak(traj, robot_params)
    if not np.isfinite(mean_abs) or mean_abs <= 1e-6:
        return spec
    target_peak = 0.995 * robot_params.wheel_speed_limit
    mean_factor = mean_abs / target_mean_abs_wheel
    peak_factor = peak / target_peak if np.isfinite(peak) else 1.0
    return _scale_spec_time(spec, max(mean_factor, peak_factor))


def generate_lemniscate(dt: float, spec: TrajectorySpec | None = None, **kwargs: Any) -> Trajectory:
    if spec is None:
        spec = TrajectorySpec(kind="lemniscate", duration=0.0, params=kwargs)
    eta = float(spec.params.get("eta", kwargs.get("eta", 1.0)))
    alpha = float(spec.params.get("alpha", kwargs.get("alpha", 4.0)))
    phase = float(spec.params.get("phase", kwargs.get("phase", 0.0)))
    x_offset = float(spec.params.get("x_offset", kwargs.get("x_offset", 0.0)))
    y_offset = float(spec.params.get("y_offset", kwargs.get("y_offset", 0.0)))
    t = np.arange(0.0, 4.0 * np.pi * alpha + dt, dt)
    tau = t + phase
    x = x_offset + eta * np.sin(tau / alpha)
    y = y_offset + eta * np.sin(tau / (2.0 * alpha))
    return _trajectory_from_xy(t, x, y, spec)


def generate_trig_combo(dt: float, spec: TrajectorySpec) -> Trajectory:
    t = np.arange(0.0, spec.duration + dt, dt)
    p = spec.params
    amps_x = np.asarray(p["amps_x"], dtype=np.float64)
    amps_y = np.asarray(p["amps_y"], dtype=np.float64)
    freqs_x = np.asarray(p["freqs_x"], dtype=np.float64)
    freqs_y = np.asarray(p["freqs_y"], dtype=np.float64)
    phases_x = np.asarray(p["phases_x"], dtype=np.float64)
    phases_y = np.asarray(p["phases_y"], dtype=np.float64)
    x = np.full_like(t, float(p.get("x_offset", 0.0)))
    y = np.full_like(t, float(p.get("y_offset", 0.0)))
    for amp, freq, phase in zip(amps_x, freqs_x, phases_x):
        x += amp * np.sin(freq * t + phase)
    for i, (amp, freq, phase) in enumerate(zip(amps_y, freqs_y, phases_y)):
        basis = np.cos if i % 2 == 0 else np.sin
        y += amp * basis(freq * t + phase)
    return _trajectory_from_xy(t, x, y, spec)


def generate_circle(dt: float, spec: TrajectorySpec) -> Trajectory:
    p = spec.params
    radius = float(p.get("radius", 0.8))
    phase = float(p.get("phase", 0.0))
    x_offset = float(p.get("x_offset", 0.0))
    y_offset = float(p.get("y_offset", 0.0))
    t = np.arange(0.0, spec.duration + dt, dt)
    tau = 2.0 * np.pi * t / spec.duration + phase
    x = x_offset + radius * np.cos(tau)
    y = y_offset + radius * np.sin(tau)
    return _trajectory_from_xy(t, x, y, spec)


def generate_ellipse(dt: float, spec: TrajectorySpec) -> Trajectory:
    p = spec.params
    a = float(p.get("a", 1.0))
    b = float(p.get("b", 0.55))
    phase = float(p.get("phase", 0.0))
    rotation = float(p.get("rotation", 0.0))
    x_offset = float(p.get("x_offset", 0.0))
    y_offset = float(p.get("y_offset", 0.0))
    t = np.arange(0.0, spec.duration + dt, dt)
    tau = 2.0 * np.pi * t / spec.duration + phase
    xr = a * np.cos(tau)
    yr = b * np.sin(tau)
    c = np.cos(rotation)
    s = np.sin(rotation)
    x = x_offset + c * xr - s * yr
    y = y_offset + s * xr + c * yr
    return _trajectory_from_xy(t, x, y, spec)


def generate_lissajous(dt: float, spec: TrajectorySpec) -> Trajectory:
    p = spec.params
    ax = float(p.get("ax", 0.8))
    ay = float(p.get("ay", 0.8))
    fx = float(p.get("fx", 2.0))
    fy = float(p.get("fy", 3.0))
    phase_x = float(p.get("phase_x", 0.0))
    phase_y = float(p.get("phase_y", np.pi / 3.0))
    x_offset = float(p.get("x_offset", 0.0))
    y_offset = float(p.get("y_offset", 0.0))
    t = np.arange(0.0, spec.duration + dt, dt)
    base = 2.0 * np.pi * t / spec.duration
    x = x_offset + ax * np.sin(fx * base + phase_x)
    y = y_offset + ay * np.sin(fy * base + phase_y)
    return _trajectory_from_xy(t, x, y, spec)


def generate_rounded_box(dt: float, spec: TrajectorySpec) -> Trajectory:
    p = spec.params
    a = float(p.get("a", 0.9))
    b = float(p.get("b", 0.65))
    exponent = float(p.get("exponent", 3.0))
    phase = float(p.get("phase", 0.0))
    rotation = float(p.get("rotation", 0.0))
    x_offset = float(p.get("x_offset", 0.0))
    y_offset = float(p.get("y_offset", 0.0))
    t = np.arange(0.0, spec.duration + dt, dt)
    tau = 2.0 * np.pi * t / spec.duration + phase
    cos_tau = np.cos(tau)
    sin_tau = np.sin(tau)
    xr = a * np.sign(cos_tau) * np.abs(cos_tau) ** (2.0 / exponent)
    yr = b * np.sign(sin_tau) * np.abs(sin_tau) ** (2.0 / exponent)
    c = np.cos(rotation)
    s = np.sin(rotation)
    x = x_offset + c * xr - s * yr
    y = y_offset + s * xr + c * yr
    return _trajectory_from_xy(t, x, y, spec)


def generate_sine_lane(dt: float, spec: TrajectorySpec) -> Trajectory:
    p = spec.params
    length = float(p.get("length", 4.0))
    amp = float(p.get("amp", 0.35))
    cycles = float(p.get("cycles", 1.5))
    phase = float(p.get("phase", 0.0))
    rotation = float(p.get("rotation", 0.0))
    x_offset = float(p.get("x_offset", 0.0))
    y_offset = float(p.get("y_offset", 0.0))
    t = np.arange(0.0, spec.duration + dt, dt)
    s_path = t / max(spec.duration, dt)
    xr = length * (s_path - 0.5)
    yr = amp * np.sin(2.0 * np.pi * cycles * s_path + phase)
    c = np.cos(rotation)
    s = np.sin(rotation)
    x = x_offset + c * xr - s * yr
    y = y_offset + s * xr + c * yr
    return _trajectory_from_xy(t, x, y, spec)


def generate_slalom(dt: float, spec: TrajectorySpec) -> Trajectory:
    p = spec.params
    length = float(p.get("length", 4.0))
    amp1 = float(p.get("amp1", 0.35))
    amp2 = float(p.get("amp2", 0.12))
    phase = float(p.get("phase", 0.0))
    rotation = float(p.get("rotation", 0.0))
    x_offset = float(p.get("x_offset", 0.0))
    y_offset = float(p.get("y_offset", 0.0))
    t = np.arange(0.0, spec.duration + dt, dt)
    s_path = t / max(spec.duration, dt)
    xr = length * (s_path - 0.5)
    yr = amp1 * np.sin(2.0 * np.pi * 2.0 * s_path + phase) + amp2 * np.sin(2.0 * np.pi * 4.0 * s_path)
    c = np.cos(rotation)
    s = np.sin(rotation)
    x = x_offset + c * xr - s * yr
    y = y_offset + s * xr + c * yr
    return _trajectory_from_xy(t, x, y, spec)


def generate_interpolated(dt: float, spec: TrajectorySpec) -> Trajectory:
    waypoints = np.asarray(spec.params["waypoints"], dtype=np.float64)
    if np.allclose(waypoints[0], waypoints[-1]):
        waypoints = waypoints[:-1]
    n_segments = len(waypoints)
    segment_steps = max(12, int(round(spec.duration / n_segments / dt)))
    xs = []
    ys = []
    for idx in range(n_segments):
        p0 = waypoints[(idx - 1) % n_segments]
        p1 = waypoints[idx % n_segments]
        p2 = waypoints[(idx + 1) % n_segments]
        p3 = waypoints[(idx + 2) % n_segments]
        s = np.linspace(0.0, 1.0, segment_steps, endpoint=False)
        s2 = s * s
        s3 = s2 * s
        pts = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * s[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * s3[:, None]
        )
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    t = np.arange(len(x), dtype=np.float64) * dt
    return _trajectory_from_xy(t, x, y, spec)


def generate_trajectory(dt: float, spec: TrajectorySpec) -> Trajectory:
    if spec.kind == "lemniscate":
        return generate_lemniscate(dt, spec)
    if spec.kind == "trig_combo":
        return generate_trig_combo(dt, spec)
    if spec.kind == "circle":
        return generate_circle(dt, spec)
    if spec.kind == "ellipse":
        return generate_ellipse(dt, spec)
    if spec.kind == "lissajous":
        return generate_lissajous(dt, spec)
    if spec.kind == "rounded_box":
        return generate_rounded_box(dt, spec)
    if spec.kind == "sine_lane":
        return generate_sine_lane(dt, spec)
    if spec.kind == "slalom":
        return generate_slalom(dt, spec)
    if spec.kind == "interpolated":
        return generate_interpolated(dt, spec)
    raise ValueError(f"Unknown trajectory kind: {spec.kind}")


def sample_trajectory_specs(
    num_runs: int,
    seed: int,
    robot_params: RobotParams,
    dt: float,
    family_order: list[str] | None = None,
    target_mean_abs_wheel_range: tuple[float, float] = (5.8, 8.8),
    min_reference_peak: float = 8.4,
    max_reference_peak_fraction: float = 0.995,
) -> list[TrajectorySpec]:
    rng = np.random.default_rng(seed)
    specs: list[TrajectorySpec] = []
    attempts = 0
    while len(specs) < num_runs:
        attempts += 1
        if attempts > 220 * max(1, num_runs):
            raise RuntimeError("Could not sample enough feasible trajectories.")

        default_family_order = [
            "lemniscate", "interpolated", "sine_lane", "slalom", "circle", "ellipse",
            "trig_combo", "lissajous", "rounded_box", "interpolated", "sine_lane", "lemniscate",
        ]
        active_family_order = default_family_order if family_order is None else family_order
        if not active_family_order:
            raise ValueError("family_order must contain at least one trajectory family.")
        family = active_family_order[len(specs) % len(active_family_order)]
        if family == "lemniscate":
            spec = TrajectorySpec(
                kind="lemniscate",
                duration=0.0,
                params={
                    "eta": float(rng.uniform(0.65, 1.45)),
                    "alpha": float(rng.uniform(1.8, 4.6)),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "x_offset": float(rng.uniform(-0.2, 0.2)),
                    "y_offset": float(rng.uniform(-0.2, 0.2)),
                },
            )
        elif family == "trig_combo":
            duration = float(rng.uniform(26.0, 62.0))
            base = 2.0 * np.pi / duration
            spec = TrajectorySpec(
                kind="trig_combo",
                duration=duration,
                params={
                    "amps_x": rng.uniform(0.18, 0.85, size=4).tolist(),
                    "amps_y": rng.uniform(0.18, 0.85, size=4).tolist(),
                    "freqs_x": (base * rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=4, replace=False)).tolist(),
                    "freqs_y": (base * rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=4, replace=False)).tolist(),
                    "phases_x": rng.uniform(0.0, 2.0 * np.pi, size=4).tolist(),
                    "phases_y": rng.uniform(0.0, 2.0 * np.pi, size=4).tolist(),
                    "x_offset": float(rng.uniform(-0.2, 0.2)),
                    "y_offset": float(rng.uniform(-0.2, 0.2)),
                },
            )
        elif family == "interpolated":
            n_wp = int(rng.integers(5, 8))
            angles = np.linspace(0.0, 2.0 * np.pi, n_wp, endpoint=False) + rng.uniform(-0.35, 0.35, size=n_wp)
            radii = rng.uniform(0.45, 1.25, size=n_wp)
            waypoints = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            offset = rng.uniform(-0.15, 0.15, size=2)
            waypoints += offset
            waypoints = np.vstack([waypoints, waypoints[0]])
            spec = TrajectorySpec(
                kind="interpolated",
                duration=float(rng.uniform(38.0, 82.0)),
                params={"waypoints": waypoints.tolist()},
            )
        elif family == "circle":
            spec = TrajectorySpec(
                kind="circle",
                duration=float(rng.uniform(22.0, 56.0)),
                params={
                    "radius": float(rng.uniform(0.45, 1.15)),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "x_offset": float(rng.uniform(-0.2, 0.2)),
                    "y_offset": float(rng.uniform(-0.2, 0.2)),
                },
            )
        elif family == "ellipse":
            spec = TrajectorySpec(
                kind="ellipse",
                duration=float(rng.uniform(26.0, 66.0)),
                params={
                    "a": float(rng.uniform(0.55, 1.35)),
                    "b": float(rng.uniform(0.35, 1.0)),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "rotation": float(rng.uniform(0.0, np.pi)),
                    "x_offset": float(rng.uniform(-0.2, 0.2)),
                    "y_offset": float(rng.uniform(-0.2, 0.2)),
                },
            )
        elif family == "lissajous":
            spec = TrajectorySpec(
                kind="lissajous",
                duration=float(rng.uniform(34.0, 78.0)),
                params={
                    "ax": float(rng.uniform(0.45, 1.0)),
                    "ay": float(rng.uniform(0.45, 1.0)),
                    "fx": float(rng.choice([1.0, 2.0, 3.0])),
                    "fy": float(rng.choice([2.0, 3.0, 4.0])),
                    "phase_x": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "phase_y": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "x_offset": float(rng.uniform(-0.15, 0.15)),
                    "y_offset": float(rng.uniform(-0.15, 0.15)),
                },
            )
        elif family == "rounded_box":
            spec = TrajectorySpec(
                kind="rounded_box",
                duration=float(rng.uniform(44.0, 92.0)),
                params={
                    "a": float(rng.uniform(0.55, 1.1)),
                    "b": float(rng.uniform(0.45, 0.9)),
                    "exponent": float(rng.uniform(2.1, 2.8)),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "rotation": float(rng.uniform(0.0, np.pi)),
                    "x_offset": float(rng.uniform(-0.15, 0.15)),
                    "y_offset": float(rng.uniform(-0.15, 0.15)),
                },
            )
        elif family == "sine_lane":
            spec = TrajectorySpec(
                kind="sine_lane",
                duration=float(rng.uniform(22.0, 52.0)),
                params={
                    "length": float(rng.uniform(3.0, 5.4)),
                    "amp": float(rng.uniform(0.18, 0.45)),
                    "cycles": float(rng.choice([1.0, 1.5, 2.0])),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "rotation": float(rng.uniform(0.0, np.pi)),
                    "x_offset": float(rng.uniform(-0.2, 0.2)),
                    "y_offset": float(rng.uniform(-0.2, 0.2)),
                },
            )
        else:
            spec = TrajectorySpec(
                kind="slalom",
                duration=float(rng.uniform(24.0, 56.0)),
                params={
                    "length": float(rng.uniform(3.2, 5.6)),
                    "amp1": float(rng.uniform(0.18, 0.42)),
                    "amp2": float(rng.uniform(0.04, 0.14)),
                    "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
                    "rotation": float(rng.uniform(0.0, np.pi)),
                    "x_offset": float(rng.uniform(-0.2, 0.2)),
                    "y_offset": float(rng.uniform(-0.2, 0.2)),
                },
            )

        target_mean_abs = float(rng.uniform(*target_mean_abs_wheel_range))
        spec = _tune_reference_speed(spec, dt, robot_params, target_mean_abs)
        traj = generate_trajectory(dt, spec)
        peak = _reference_wheel_peak(traj, robot_params)
        mean_abs = _reference_wheel_mean_abs(traj, robot_params)
        min_mean_abs_by_kind = {
            "circle": 7.8,
            "ellipse": 6.8,
            "sine_lane": 6.6,
            "slalom": 6.4,
            "trig_combo": 4.2,
            "lemniscate": 3.2,
            "lissajous": 3.2,
            "interpolated": 2.8,
            "rounded_box": 2.8,
        }
        if (
            reference_is_feasible(traj, robot_params, margin=max_reference_peak_fraction)
            and peak >= min_reference_peak
            and mean_abs >= min(min_mean_abs_by_kind[spec.kind], target_mean_abs_wheel_range[0])
        ):
            specs.append(spec)
    return specs
