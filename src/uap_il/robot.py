from __future__ import annotations

import numpy as np

from .config import RobotParams


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wheel_to_unicycle(wr: float, wl: float, params: RobotParams) -> tuple[float, float]:
    v = params.wheel_radius * (wr + wl) / 2.0
    omega = params.wheel_radius * (wr - wl) / params.wheelbase
    return v, omega


def unicycle_to_wheel(v: float, omega: float, params: RobotParams) -> tuple[float, float]:
    wr = (v + 0.5 * params.wheelbase * omega) / params.wheel_radius
    wl = (v - 0.5 * params.wheelbase * omega) / params.wheel_radius
    return wr, wl


def clip_wheels(wr: float, wl: float, params: RobotParams) -> tuple[float, float]:
    limit = params.wheel_speed_limit
    return float(np.clip(wr, -limit, limit)), float(np.clip(wl, -limit, limit))


def clip_unicycle_to_feasible(v: float, omega: float, params: RobotParams) -> tuple[float, float, float, float]:
    wr, wl = unicycle_to_wheel(v, omega, params)
    wr, wl = clip_wheels(wr, wl, params)
    v, omega = wheel_to_unicycle(wr, wl, params)
    return v, omega, wr, wl


def step_differential_drive(
    state: np.ndarray,
    wr: float,
    wl: float,
    dt: float,
    params: RobotParams,
) -> np.ndarray:
    wr, wl = clip_wheels(wr, wl, params)
    v, omega = wheel_to_unicycle(wr, wl, params)
    x, y, theta = state
    next_state = np.array(
        [
            x + dt * v * np.cos(theta),
            y + dt * v * np.sin(theta),
            wrap_to_pi(theta + dt * omega),
        ],
        dtype=np.float64,
    )
    return next_state
