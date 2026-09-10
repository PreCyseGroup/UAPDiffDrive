from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DeLucaGains, RobotParams
from .robot import clip_unicycle_to_feasible
from .trajectories import Trajectory


@dataclass
class ControllerState:
    v_curr: float


def deluca_controller(
    state: np.ndarray,
    k: int,
    trajectory: Trajectory,
    controller_state: ControllerState,
    gains: DeLucaGains,
    robot_params: RobotParams,
    dt: float,
) -> tuple[float, float, ControllerState]:
    x, y, theta = state
    xd_curr = controller_state.v_curr * np.cos(theta)
    yd_curr = controller_state.v_curr * np.sin(theta)

    u1 = trajectory.xdd[k] + gains.kp1 * (trajectory.x[k] - x) + gains.kd1 * (trajectory.xd[k] - xd_curr)
    u2 = trajectory.ydd[k] + gains.kp2 * (trajectory.y[k] - y) + gains.kd2 * (trajectory.yd[k] - yd_curr)

    a = u1 * np.cos(theta) + u2 * np.sin(theta)
    v = controller_state.v_curr + a * dt
    if abs(v) < 1e-4:
        omega = 0.0
    else:
        omega = (u2 * np.cos(theta) - u1 * np.sin(theta)) / v

    v, omega, wr, wl = clip_unicycle_to_feasible(v, omega, robot_params)
    return wr, wl, ControllerState(v_curr=v)
