from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotParams:
    wheel_radius: float = 0.04445
    wheelbase: float = 0.393
    wheel_speed_limit: float = 10.0


@dataclass(frozen=True)
class SimParams:
    dt: float = 1.0 / 60.0
    horizon_scale: float = 1.0
    lookahead_time: float = 0.35


@dataclass(frozen=True)
class DeLucaGains:
    kp1: float = 3.0
    kp2: float = 3.0
    kd1: float = 1.5
    kd2: float = 1.5


@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    plot_dir: Path = Path("plots")

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
