from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


POLICY_INPUT_DIM = 8


@dataclass(frozen=True)
class RobotParams:
    wheel_radius: float = 0.04445
    wheelbase: float = 0.393
    wheel_speed_limit: float = 10.0


class ThetaTrigPreprocessor(nn.Module):
    """Append cos(theta_raw), sin(theta_raw) to normalized inputs."""

    def __init__(
        self,
        input_dim: int,
        theta_index: int = 2,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(input_dim + 2)
        self.theta_index = int(theta_index)
        if x_mean is None:
            x_mean = torch.zeros(input_dim, dtype=torch.float32)
        if x_std is None:
            x_std = torch.ones(input_dim, dtype=torch.float32)
        self.register_buffer("x_mean", x_mean.detach().clone().float())
        self.register_buffer("x_std", x_std.detach().clone().float())

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        theta_raw = x_norm[..., self.theta_index] * self.x_std[self.theta_index] + self.x_mean[self.theta_index]
        return torch.cat(
            [
                x_norm,
                torch.cos(theta_raw).unsqueeze(-1),
                torch.sin(theta_raw).unsqueeze(-1),
            ],
            dim=-1,
        )


def _n_groups(dim: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if dim % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(_n_groups(dim), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.GroupNorm(_n_groups(dim), dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TrajectoryMLP(nn.Module):
    def __init__(self, n_inputs: int = 10, n_outputs: int = 2, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(n_inputs, 512), nn.GroupNorm(_n_groups(512), 512), nn.SiLU())
        self.res_block_1a = ResidualBlock(512, dropout)
        self.res_block_1b = ResidualBlock(512, dropout)
        self.down1 = nn.Sequential(nn.Linear(512, 256), nn.GroupNorm(_n_groups(256), 256), nn.SiLU())
        self.res_block_2a = ResidualBlock(256, dropout)
        self.res_block_2b = ResidualBlock(256, dropout)
        self.down2 = nn.Sequential(nn.Linear(256, 128), nn.GroupNorm(_n_groups(128), 128), nn.SiLU())
        self.head = nn.Linear(128, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_block_1a(x)
        x = self.res_block_1b(x)
        x = self.down1(x)
        x = self.res_block_2a(x)
        x = self.res_block_2b(x)
        x = self.down2(x)
        return self.head(x)


class WheelMLP(nn.Module):
    """Residual trajectory policy matching the saved imitation-learning checkpoint."""

    def __init__(
        self,
        input_dim: int = POLICY_INPUT_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        theta_index: int = 2,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.preprocess = ThetaTrigPreprocessor(input_dim, theta_index=theta_index, x_mean=x_mean, x_std=x_std)
        internal_dim = self.preprocess.output_dim
        self.net = TrajectoryMLP(n_inputs=internal_dim, n_outputs=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.preprocess(x))


class TrainedWheelPolicy:
    def __init__(self, checkpoint_path: str | Path, norm_stats_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        stats_npz = np.load(norm_stats_path)
        self.stats = {key: stats_npz[key].astype(np.float32) for key in stats_npz.files}
        input_dim = int(self.stats["x_mean"].shape[0])
        if input_dim != POLICY_INPUT_DIM:
            raise RuntimeError(f"Expected {POLICY_INPUT_DIM} policy inputs, got {input_dim}.")
        self.model = WheelMLP(
            input_dim=input_dim,
            x_mean=torch.from_numpy(self.stats["x_mean"]),
            x_std=torch.from_numpy(self.stats["x_std"]),
        ).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def wheel_speeds(self, feature_raw: np.ndarray, robot: RobotParams) -> tuple[float, float]:
        feature = np.asarray(feature_raw, dtype=np.float32)
        if feature.shape != (POLICY_INPUT_DIM,):
            raise ValueError(f"Policy feature has shape {feature.shape}; expected ({POLICY_INPUT_DIM},).")
        x_norm = (feature - self.stats["x_mean"]) / self.stats["x_std"]
        with torch.no_grad():
            y_norm = self.model(torch.from_numpy(x_norm[None, :]).to(self.device)).detach().cpu().numpy()[0]
        wr, wl = y_norm * self.stats["y_std"] + self.stats["y_mean"]
        return clip_wheels(float(wr), float(wl), robot)


def clip_wheels(wr: float, wl: float, robot: RobotParams) -> tuple[float, float]:
    limit = robot.wheel_speed_limit
    return float(np.clip(wr, -limit, limit)), float(np.clip(wl, -limit, limit))


def wheel_to_unicycle(wr: float, wl: float, robot: RobotParams) -> tuple[float, float]:
    v = robot.wheel_radius * (wr + wl) / 2.0
    omega = robot.wheel_radius * (wr - wl) / robot.wheelbase
    return float(v), float(omega)


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
