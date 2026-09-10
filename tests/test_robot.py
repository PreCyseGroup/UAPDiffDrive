import numpy as np

from uap_il.config import RobotParams
from uap_il.robot import clip_wheels, step_differential_drive, unicycle_to_wheel, wheel_to_unicycle


def test_wheel_unicycle_round_trip():
    params = RobotParams()
    v, omega = 0.25, 0.4
    wr, wl = unicycle_to_wheel(v, omega, params)
    v_back, omega_back = wheel_to_unicycle(wr, wl, params)
    assert np.isclose(v, v_back)
    assert np.isclose(omega, omega_back)


def test_clip_wheels_respects_robot_limit():
    params = RobotParams(wheel_speed_limit=10.0)
    wr, wl = clip_wheels(12.0, -14.0, params)
    assert wr == 10.0
    assert wl == -10.0


def test_step_differential_drive_moves_forward():
    params = RobotParams()
    state = np.array([0.0, 0.0, 0.0])
    next_state = step_differential_drive(state, 5.0, 5.0, 0.1, params)
    assert next_state[0] > 0.0
    assert np.isclose(next_state[1], 0.0)
