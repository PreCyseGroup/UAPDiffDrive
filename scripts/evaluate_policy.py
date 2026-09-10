#!/usr/bin/env python
"""Evaluate a checkpoint on independent references; optionally fit a fixed UAP.

UAP fitting uses alpha=4, phase=0. Evaluation uses alpha=5, phase=0.7.
Outputs are created only when invoked and are not part of the source release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uap_il.attack import learn_closed_loop_uap
from uap_il.settings import load_settings
from uap_il.simulate import compute_metrics, rollout_policy
from uap_il.train import load_policy
from uap_il.trajectories import TrajectorySpec, generate_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sim2real_nominal_policy.pt"))
    parser.add_argument("--norm", type=Path, default=Path("models/sim2real_nominal_norm_stats.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--uap", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--attack-steps", type=int, default=25)
    parser.add_argument("--attack-horizon", type=int, default=900)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.epsilon < 0 or args.attack_steps < 1 or args.attack_horizon < 2 or args.threads < 1:
        parser.error("epsilon must be nonnegative; steps/threads positive; horizon >= 2")
    torch.set_num_threads(args.threads)
    settings = load_settings(args.config)
    model, stats = load_policy(args.checkpoint, args.norm, device="cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def reference(alpha, phase):
        trajectory = generate_trajectory(settings.simulation.dt, TrajectorySpec(
            kind="lemniscate", duration=0.0,
            params={"eta": 1.0, "alpha": alpha, "phase": phase}))
        initial = np.array([trajectory.x[0] + 0.1, trajectory.y[0], trajectory.theta[0]])
        return trajectory, initial
    test, initial = reference(5.0, 0.7)
    clean = rollout_policy(model, stats, test, initial, settings.robot, settings.simulation)
    report = {"checkpoint": str(args.checkpoint), "norm": str(args.norm),
              "clean": compute_metrics(clean), "evaluation_reference": {"alpha": 5.0, "phase": 0.7}}
    np.savez_compressed(args.output_dir / "clean.npz", **clean)
    if args.uap:
        fit, fit_initial = reference(4.0, 0.0)
        mask = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        delta, normalized, metadata = learn_closed_loop_uap(
            model, stats, [fit], [fit_initial], settings.robot, settings.simulation,
            epsilon=args.epsilon, steps=args.attack_steps, mask=mask,
            horizon_steps=args.attack_horizon, optimizer="gradient", device="cpu")
        assert np.isfinite(delta).all()
        assert np.max(np.abs(normalized)) <= args.epsilon + 1e-6
        np.testing.assert_array_equal(normalized[3:], np.zeros(5))
        attacked = rollout_policy(model, stats, test, initial, settings.robot,
                                  settings.simulation, perturbation=delta)
        np.save(args.output_dir / "uap_raw.npy", delta)
        np.save(args.output_dir / "uap_normalized.npy", normalized)
        np.savez_compressed(args.output_dir / "attacked.npz", **attacked)
        report.update(attacked=compute_metrics(attacked), epsilon=args.epsilon,
                      attack=metadata, attack_reference={"alpha": 4.0, "phase": 0.0})
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
