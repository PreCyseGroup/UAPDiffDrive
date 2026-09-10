#!/usr/bin/env python
"""Evaluate baseline and residual adversarial policies against UAP and FGSM on the wall track."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-uap-il")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import interpolated_wall_showcase as wall
from uap_il.config import Paths
from uap_il.device import resolve_device
from uap_il.settings import load_settings
from uap_il.train import load_policy


COLORS = {
    "baseline_clean": "#0072B2",
    "baseline_uap": "#D55E00",
    "baseline_fgsm": "#E69F00",
    "defense_clean": "#009E73",
    "defense_uap": "#CC0000",
    "defense_fgsm": "#8A3FFC",
}


def epsilon_tag(epsilon: float) -> str:
    return f"{epsilon:.3f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolated-wall robustness sweep for residual defense policy.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpolated_wall_residual_robustness"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("artifacts/il_policy.pt"))
    parser.add_argument("--baseline-norm", type=Path, default=Path("artifacts/norm_stats.npz"))
    parser.add_argument("--defense-checkpoint", type=Path, default=Path("artifacts/residual_adversarial_defense_policy.pt"))
    parser.add_argument("--defense-norm", type=Path, default=Path("artifacts/residual_adversarial_defense_norm_stats.npz"))
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.10, 0.25, 0.35, 0.50])
    parser.add_argument("--uap-epsilons", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--robot-radius", type=float, default=0.18)
    parser.add_argument("--track-width", type=float, default=0.70)
    parser.add_argument("--attack-start", type=float, default=14.0)
    parser.add_argument("--attack-end", type=float, default=20.0)
    parser.add_argument("--fgsm-target-mode", choices=["reverse_output", "zero_output"], default="reverse_output")
    parser.add_argument("--min-attack-forward-fraction", type=float, default=0.35)
    return parser.parse_args()


def _load_raw_uap(path: Path, stats: dict[str, np.ndarray]) -> np.ndarray:
    delta = np.load(path).astype(np.float32)
    if delta.shape != stats["x_std"].shape:
        raise ValueError(f"{path} has shape {delta.shape}, expected {stats['x_std'].shape}.")
    return delta


def _remap_uap(
    baseline_delta: np.ndarray,
    baseline_stats: dict[str, np.ndarray],
    target_stats: dict[str, np.ndarray],
) -> np.ndarray:
    normalized = baseline_delta / baseline_stats["x_std"].astype(np.float32)
    return normalized * target_stats["x_std"].astype(np.float32)


def _case_metrics(
    policy_name: str,
    attack_name: str,
    epsilon: float | None,
    rollout: dict,
    clean_metrics: dict[str, object],
    track_width: float,
    robot_radius: float,
    mask: np.ndarray,
) -> dict[str, object]:
    metrics = wall.rollout_metrics(rollout, track_width, robot_radius)
    return {
        "policy": policy_name,
        "attack": attack_name,
        "epsilon": "" if epsilon is None else float(epsilon),
        **metrics,
        "rmse_increase_vs_clean": metrics["rmse_pos"] - float(clean_metrics["rmse_pos"]),
        "clearance_decrease_vs_clean": float(clean_metrics["min_wall_clearance"]) - metrics["min_wall_clearance"],
        "normalized_linf_bound": wall.normalized_bound(rollout, mask),
    }


def _save_metric_plots(out_dir: Path, rows: list[dict[str, object]]) -> None:
    attack_rows = [row for row in rows if row["attack"] != "clean"]
    labels = [f"{row['policy']}\n{row['attack']} {row['epsilon']}" for row in attack_rows]
    rmse = [float(row["rmse_pos"]) for row in attack_rows]
    clearance = [float(row["min_wall_clearance"]) for row in attack_rows]
    colors = [COLORS[f"{row['policy']}_{row['attack']}"] for row in attack_rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.72 * len(labels)), 8.2), sharex=True)
    x = np.arange(len(labels))
    axes[0].bar(x, rmse, color=colors)
    axes[0].set_ylabel("RMSE [m]")
    axes[0].set_title("Interpolated-Wall Attack Robustness")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, clearance, color=colors)
    axes[1].axhline(0.0, color="#222222", lw=1.0)
    axes[1].set_ylabel("min clearance [m]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_bars.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "robustness_bars.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_top_down(out_dir: Path, trajectory, selected: dict[str, dict], args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    wall.draw_track(ax, trajectory, args.track_width, args.robot_radius)
    wall.draw_attack_window(ax, trajectory, args.attack_start, args.attack_end)
    for name, rollout in selected.items():
        color = COLORS[name]
        ax.plot(rollout["states"][:, 0], rollout["states"][:, 1], color=color, lw=2.3, label=name.replace("_", " "))
        if rollout["collision"]:
            idx = int(rollout["collision_index"])
            ax.scatter(
                [rollout["states"][idx, 0]],
                [rollout["states"][idx, 1]],
                s=130,
                facecolors="none",
                edgecolors="#C23B55",
                linewidths=2.2,
                zorder=9,
            )
    ax.set_title("Interpolated-Wall Robustness: eps=0.25")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", ncol=2, frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "top_down_eps_0p250.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "top_down_eps_0p250.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_trajectories(out_dir: Path, rollouts: dict[str, dict]) -> None:
    trajectory_dir = out_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, rollout in sorted(rollouts.items()):
        npz_path = trajectory_dir / f"{name}.npz"
        csv_path = trajectory_dir / f"{name}.csv"
        np.savez_compressed(
            npz_path,
            t=rollout["t"],
            states=rollout["states"],
            refs=rollout["refs"],
            features=rollout["features"],
            wheels=rollout["wheels"],
            perturbations=rollout["perturbations"],
            perturbations_normalized=rollout["perturbations_normalized"],
        )
        table = np.column_stack([rollout["t"], rollout["refs"], rollout["states"], rollout["wheels"]])
        np.savetxt(
            csv_path,
            table,
            delimiter=",",
            header="t,ref_x,ref_y,ref_theta,state_x,state_y,state_theta,wheel_right,wheel_left",
            comments="",
        )
        manifest.append(
            {
                "name": name,
                "npz": str(npz_path),
                "csv": str(csv_path),
                "steps": int(len(rollout["t"])),
                "collision": bool(rollout["collision"]),
                "collision_time": rollout["collision_time"],
            }
        )
    (trajectory_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    device = resolve_device(args.device or settings.device)
    paths = Paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wall.setup_plot_style()

    strategy = args.strategy or ("ex_ey_theta" if "ex_ey_theta" in settings.attack_strategies else next(iter(settings.attack_strategies)))
    mask = np.asarray(settings.attack_strategies[strategy], dtype=np.float32)
    trajectory = wall.make_interpolated_showcase(settings.simulation.dt)
    initial_state = np.array([trajectory.x[0], trajectory.y[0], trajectory.theta[0]], dtype=np.float64)
    attack_end = min(float(args.attack_end), float(trajectory.t[-1]))

    baseline_model, baseline_stats = load_policy(args.baseline_checkpoint, args.baseline_norm, device=device)
    defense_model, defense_stats = load_policy(args.defense_checkpoint, args.defense_norm, device=device)
    policies = {
        "baseline": (baseline_model, baseline_stats),
        "defense": (defense_model, defense_stats),
    }

    clean_rollouts: dict[str, dict] = {}
    clean_metrics: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    selected_top_down: dict[str, dict] = {}
    all_rollouts: dict[str, dict] = {}

    for policy_name, (model, stats) in policies.items():
        clean = wall.rollout_with_attack_and_walls(
            model,
            stats,
            trajectory,
            initial_state,
            settings.robot,
            settings.simulation,
            "no_attack",
            0.0,
            mask,
            args.track_width,
            args.robot_radius,
            args.attack_start,
            attack_end,
        )
        clean_rollouts[policy_name] = clean
        clean_metrics[policy_name] = wall.rollout_metrics(clean, args.track_width, args.robot_radius)
        rows.append(
            {
                "policy": policy_name,
                "attack": "clean",
                "epsilon": "",
                **clean_metrics[policy_name],
                "rmse_increase_vs_clean": 0.0,
                "clearance_decrease_vs_clean": 0.0,
                "normalized_linf_bound": 0.0,
            }
        )
        selected_top_down[f"{policy_name}_clean"] = clean
        all_rollouts[f"{policy_name}_clean"] = clean

    missing_uap: list[str] = []
    for epsilon in sorted({float(value) for value in args.uap_epsilons}):
        uap_path = paths.artifact_dir / f"uap_{strategy}_eps_{epsilon_tag(epsilon)}.npy"
        if not uap_path.exists():
            missing_uap.append(str(uap_path))
            continue
        baseline_uap = _load_raw_uap(uap_path, baseline_stats)
        for policy_name, (model, stats) in policies.items():
            delta = baseline_uap if policy_name == "baseline" else _remap_uap(baseline_uap, baseline_stats, stats)
            rollout = wall.rollout_with_attack_and_walls(
                model,
                stats,
                trajectory,
                initial_state,
                settings.robot,
                settings.simulation,
                "uap",
                epsilon,
                mask,
                args.track_width,
                args.robot_radius,
                args.attack_start,
                attack_end,
                uap_delta=delta,
                min_attack_forward_fraction=args.min_attack_forward_fraction,
            )
            rows.append(
                _case_metrics(
                    policy_name,
                    "uap",
                    epsilon,
                    rollout,
                    clean_metrics[policy_name],
                    args.track_width,
                    args.robot_radius,
                    mask,
                )
            )
            if abs(epsilon - 0.25) < 1e-9:
                selected_top_down[f"{policy_name}_uap"] = rollout
            all_rollouts[f"{policy_name}_uap_eps_{epsilon_tag(epsilon)}"] = rollout

    for epsilon in sorted({float(value) for value in args.epsilons}):
        for policy_name, (model, stats) in policies.items():
            rollout = wall.rollout_with_attack_and_walls(
                model,
                stats,
                trajectory,
                initial_state,
                settings.robot,
                settings.simulation,
                "fgsm",
                epsilon,
                mask,
                args.track_width,
                args.robot_radius,
                args.attack_start,
                attack_end,
                fgsm_target_mode=args.fgsm_target_mode,
                min_attack_forward_fraction=args.min_attack_forward_fraction,
            )
            rows.append(
                _case_metrics(
                    policy_name,
                    "fgsm",
                    epsilon,
                    rollout,
                    clean_metrics[policy_name],
                    args.track_width,
                    args.robot_radius,
                    mask,
                )
            )
            if abs(epsilon - 0.25) < 1e-9:
                selected_top_down[f"{policy_name}_fgsm"] = rollout
            all_rollouts[f"{policy_name}_fgsm_eps_{epsilon_tag(epsilon)}"] = rollout

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "scenario": "interpolated_wall_residual_robustness",
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "defense_checkpoint": str(args.defense_checkpoint),
        "baseline_norm": str(args.baseline_norm),
        "defense_norm": str(args.defense_norm),
        "strategy": strategy,
        "mask": mask.tolist(),
        "attack_start": float(args.attack_start),
        "attack_end": float(attack_end),
        "fgsm_target_mode": args.fgsm_target_mode,
        "min_attack_forward_fraction": float(args.min_attack_forward_fraction),
        "missing_uap_files": missing_uap,
        "results": rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Interpolated-Wall Residual Robustness Evaluation\n\n"
        "This evaluation compares the baseline residual NN and the residual adversarial-defense NN on the "
        "interpolated-wall trajectory under clean, saved UAP, and online FGSM perturbations. UAP perturbations "
        "are loaded from the baseline artifact files and remapped through the defense normalization statistics "
        "when needed; FGSM is computed online from the same policy under test.\n",
        encoding="utf-8",
    )
    _save_metric_plots(args.output_dir, rows)
    _save_top_down(args.output_dir, trajectory, selected_top_down, args)
    _save_trajectories(args.output_dir, all_rollouts)

    for row in rows:
        eps = "" if row["epsilon"] == "" else f" eps={float(row['epsilon']):.3f}"
        print(
            f"[{row['policy']} {row['attack']}{eps}] "
            f"rmse={float(row['rmse_pos']):.4f} clearance={float(row['min_wall_clearance']):.4f} "
            f"collision={row['collision']}",
            flush=True,
        )
    print(f"[done] wrote outputs -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
