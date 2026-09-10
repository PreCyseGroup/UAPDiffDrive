from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DeLucaGains, RobotParams, SimParams
from .dataset import POLICY_INPUT_DIM

DEFAULT_ATTACK_STRATEGIES = {
    "ex_ey_theta": [1.0, 1.0, 1.0] + [0.0] * (POLICY_INPUT_DIM - 3),
}


@dataclass(frozen=True)
class WorkflowSettings:
    num_runs: int
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    max_train_samples: int
    max_val_samples: int
    dataset_state_noise: dict[str, float]
    device: str
    attack_epsilon: float
    attack_epsilons: list[float]
    attack_strategies: dict[str, list[float]]
    attack_method: str
    attack_steps: int
    attack_batch_size: int
    attack_population_size: int
    attack_horizon_steps: int
    attack_score_mode: str
    gaussian_noise_stds: list[float]
    gaussian_noise_repeats: int
    evaluation: dict[str, float]
    robot: RobotParams
    simulation: SimParams
    deluca_gains: DeLucaGains


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(config_path: Path, overrides: dict[str, Any] | None = None) -> WorkflowSettings:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if overrides:
        raw = _merge_dict(raw, {key: value for key, value in overrides.items() if value is not None})
    robot = RobotParams(**raw["robot"])
    simulation = SimParams(**raw["simulation"])
    gains = DeLucaGains(**raw["deluca_gains"])
    return WorkflowSettings(
        num_runs=int(raw["num_runs"]),
        seed=int(raw["seed"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        learning_rate=float(raw["learning_rate"]),
        max_train_samples=int(raw.get("max_train_samples", 0)),
        max_val_samples=int(raw.get("max_val_samples", 0)),
        dataset_state_noise={key: float(value) for key, value in raw.get("dataset_state_noise", {}).items()},
        device=str(raw.get("device", "auto")),
        attack_epsilon=float(raw["attack_epsilon"]),
        attack_epsilons=[float(value) for value in raw.get("attack_epsilons", [raw["attack_epsilon"]])],
        attack_strategies={
            name: [float(value) for value in mask]
            for name, mask in raw.get("attack_strategies", DEFAULT_ATTACK_STRATEGIES).items()
        },
        attack_method=str(raw.get("attack_method", "closed_loop_gradient")),
        attack_steps=int(raw["attack_steps"]),
        attack_batch_size=int(raw["attack_batch_size"]),
        attack_population_size=int(raw.get("attack_population_size", 24)),
        attack_horizon_steps=int(raw.get("attack_horizon_steps", 900)),
        attack_score_mode=str(raw.get("attack_score_mode", "mean")),
        gaussian_noise_stds=[float(value) for value in raw.get("gaussian_noise_stds", [])],
        gaussian_noise_repeats=int(raw.get("gaussian_noise_repeats", 1)),
        evaluation={key: float(value) for key, value in raw.get("evaluation", {}).items()},
        robot=robot,
        simulation=simulation,
        deluca_gains=gains,
    )
