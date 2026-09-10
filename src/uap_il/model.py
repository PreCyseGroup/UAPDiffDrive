from __future__ import annotations

import torch
from torch import nn

from .dataset import POLICY_INPUT_DIM


class ThetaTrigPreprocessor(nn.Module):
    """Fixed layer that appends cos(theta_raw), sin(theta_raw) to normalized inputs."""

    def __init__(
        self,
        input_dim: int,
        theta_index: int = 2,
        x_mean: torch.Tensor | None = None,
        x_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= theta_index < input_dim:
            raise ValueError(f"theta_index={theta_index} is outside input_dim={input_dim}.")
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
    """Residual trajectory policy with fixed theta-to-trig preprocessing."""

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


class NormalizedPolicy(nn.Module):
    def __init__(
        self,
        model: WheelMLP,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        y_norm = self.model((x_raw - self.x_mean) / self.x_std)
        return y_norm * self.y_std + self.y_mean
