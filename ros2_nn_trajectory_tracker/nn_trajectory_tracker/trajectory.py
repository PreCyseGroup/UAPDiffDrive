from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import wrap_to_pi


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


def generate_lemniscate(
    dt: float,
    eta: float,
    alpha: float,
    phase: float,
    x_offset: float,
    y_offset: float,
) -> Trajectory:
    t = np.arange(0.0, 4.0 * np.pi * alpha + dt, dt)
    tau = t + phase
    x = x_offset + eta * np.sin(tau / alpha)
    y = y_offset + eta * np.sin(tau / (2.0 * alpha))
    xd = np.gradient(x, t, edge_order=2)
    yd = np.gradient(y, t, edge_order=2)
    xdd = np.gradient(xd, t, edge_order=2)
    ydd = np.gradient(yd, t, edge_order=2)
    theta = np.unwrap(np.arctan2(yd, xd))
    v = np.sqrt(xd**2 + yd**2)
    return Trajectory(t=t, x=x, y=y, xd=xd, yd=yd, xdd=xdd, ydd=ydd, theta=theta, v=v)


def unwrap_heading_to_reference(theta: float, reference_theta: float) -> float:
    return float(reference_theta + wrap_to_pi(theta - reference_theta))


def tracking_features(
    x: float,
    y: float,
    yaw: float,
    trajectory: Trajectory,
    k: int,
    lookahead_steps: int,
) -> np.ndarray:
    feature_k = min(k + lookahead_steps, len(trajectory.t) - 1)
    dx = trajectory.x[feature_k] - x
    dy = trajectory.y[feature_k] - y
    ex = np.cos(yaw) * dx + np.sin(yaw) * dy
    ey = -np.sin(yaw) * dx + np.cos(yaw) * dy
    theta = unwrap_heading_to_reference(yaw, float(trajectory.theta[k]))
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
