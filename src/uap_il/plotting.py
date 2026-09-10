from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .dataset import FEATURE_LABELS


def _display_label(label: str) -> str:
    return label.replace("cos_theta", "cos(theta)").replace("sin_theta", "sin(theta)")


def _field_label(label: str) -> str:
    return label.replace("cos_theta", "cos_theta").replace("sin_theta", "sin_theta")


def _format_delta(values: np.ndarray) -> str:
    labels = FEATURE_LABELS[: len(values)]
    return ", ".join(f"{_display_label(label)}={value:+.4f}" for label, value in zip(labels, values))


def save_dataset_plot(inputs: np.ndarray, targets: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(inputs[:, 0], inputs[:, 1], s=2, alpha=0.25)
    axes[0].set_title("Expert Error States")
    axes[0].set_xlabel("ex [m]")
    axes[0].set_ylabel("ey [m]")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(targets[:, 0], bins=60, alpha=0.7, label="wr")
    axes[1].hist(targets[:, 1], bins=60, alpha=0.7, label="wl")
    axes[1].set_title("Expert Wheel Speeds")
    axes[1].set_xlabel("rad/s")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_training_plot(history: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on normalized targets")
    ax.set_title("MLP Training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_trajectory_family_overview(trajectories: list, path: Path) -> None:
    seen = set()
    representatives = []
    for trajectory in trajectories:
        kind = trajectory.spec.kind
        if kind not in seen:
            seen.add(kind)
            representatives.append(trajectory)
    n = len(representatives)
    if n == 0:
        return
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, trajectory in zip(axes.ravel(), representatives):
        ax.axis("on")
        ax.plot(trajectory.x, trajectory.y, linewidth=1.8)
        ax.scatter(trajectory.x[0], trajectory.y[0], s=24, c="green", label="start")
        ax.set_title(trajectory.spec.kind)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Training Trajectory Families", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _set_bounds_from_reference(ax, refs: np.ndarray, pad_fraction: float = 0.12) -> None:
    x_min, x_max = float(refs[:, 0].min()), float(refs[:, 0].max())
    y_min, y_max = float(refs[:, 1].min()), float(refs[:, 1].max())
    width = max(x_max - x_min, 0.5)
    height = max(y_max - y_min, 0.5)
    pad = pad_fraction * max(width, height)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")


def save_rollout_plots(
    clean: dict[str, np.ndarray],
    attacked: dict[str, np.ndarray],
    out_dir: Path,
    prefix: str = "rollout",
    attacked_label: str = "IL + UAP",
    delta: list[float] | np.ndarray | None = None,
    annotation: str | None = None,
) -> None:
    t = clean["t"]
    delta_text = ""
    if delta is not None:
        values = np.asarray(delta, dtype=float)
        delta_text = "Raw UAP delta added before normalization: " + _format_delta(values)
    if annotation is not None:
        delta_text = annotation

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(clean["refs"][:, 0], clean["refs"][:, 1], "k--", label="reference")
    ax.plot(clean["states"][:, 0], clean["states"][:, 1], label="IL clean")
    ax.plot(attacked["states"][:, 0], attacked["states"][:, 1], label=attacked_label)
    ax.scatter(clean["states"][0, 0], clean["states"][0, 1], c="green", s=35, label="start")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Closed-Loop Trajectory: Reference Scale")
    if delta_text:
        fig.suptitle(delta_text, fontsize=9)
    _set_bounds_from_reference(ax, clean["refs"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.94) if delta_text else None)
    fig.savefig(out_dir / f"{prefix}_trajectory_reference_scale.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(clean["refs"][:, 0], clean["refs"][:, 1], "k--", label="reference")
    ax.plot(clean["states"][:, 0], clean["states"][:, 1], label="IL clean")
    ax.plot(attacked["states"][:, 0], attacked["states"][:, 1], label=attacked_label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Closed-Loop Trajectory: Full Extent")
    if delta_text:
        fig.suptitle(delta_text, fontsize=9)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.94) if delta_text else None)
    fig.savefig(out_dir / f"{prefix}_trajectory_full_extent.png", dpi=180)
    plt.close(fig)

    clean_ep = np.linalg.norm(clean["refs"][:, :2] - clean["states"][:, :2], axis=1)
    attacked_ep = np.linalg.norm(attacked["refs"][:, :2] - attacked["states"][:, :2], axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, clean_ep, label="IL clean")
    ax.plot(t, attacked_ep, label=attacked_label)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("position error [m]")
    ax.set_title("Tracking Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_errors.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, clean["wheels"][:, 0], label="clean wr")
    axes[0].plot(t, attacked["wheels"][:, 0], label="attack wr", alpha=0.8)
    axes[1].plot(t, clean["wheels"][:, 1], label="clean wl")
    axes[1].plot(t, attacked["wheels"][:, 1], label="attack wl", alpha=0.8)
    for ax in axes:
        ax.set_ylabel("rad/s")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[1].set_xlabel("t [s]")
    fig.suptitle("Applied Wheel Speeds")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_wheels.png", dpi=180)
    plt.close(fig)


def save_attack_summary(results: list[dict], path: Path) -> None:
    labels = [f"{row['strategy']} eps={row['epsilon']}" for row in results]
    clean = [row["clean"]["rmse_pos"] for row in results]
    attacked = [row["attacked"]["rmse_pos"] for row in results]
    mean_delta = [attacked_value - clean_value for clean_value, attacked_value in zip(clean, attacked)]
    worst_delta = []
    for row in results:
        rollout_deltas = [
            rollout["attacked"]["rmse_pos"] - rollout["clean"]["rmse_pos"]
            for rollout in row.get("per_rollout", [])
        ]
        worst_delta.append(max(rollout_deltas) if rollout_deltas else row["attacked"]["rmse_pos"] - row["clean"]["rmse_pos"])
    x = np.arange(len(results))
    fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.65 * len(results)), 7.2), sharex=True)

    axes[0].bar(x - 0.18, clean, width=0.36, label="clean")
    axes[0].bar(x + 0.18, attacked, width=0.36, label="attacked")
    axes[0].set_ylabel("position RMSE [m]")
    axes[0].set_title("Attack Strategy and Epsilon Sweep")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].bar(x - 0.18, mean_delta, width=0.36, label="mean attacked - clean")
    axes[1].bar(x + 0.18, worst_delta, width=0.36, label="worst rollout attacked - clean")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("RMSE increase [m]")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_attack_noise_comparison_summary(
    attack_results: list[dict],
    gaussian_results: list[dict],
    path: Path,
) -> None:
    labels = []
    values = []
    colors = []
    for row in sorted(attack_results, key=lambda item: (item["strategy"], item["epsilon"])):
        labels.append(f"UAP {row['strategy']} {row['epsilon']}")
        values.append(row["attacked"]["rmse_pos"])
        colors.append("#c44e52")
    for row in sorted(gaussian_results, key=lambda item: (item["strategy"], item["std_normalized"])):
        labels.append(f"Gaussian {row['strategy']} {row['std_normalized']}")
        values.append(row["attacked"]["rmse_pos"])
        colors.append("#4c72b0")
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(labels)), 5.2))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("position RMSE [m]")
    ax.set_title("Optimized UAP vs Gaussian Random Noise")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_animation(clean: dict[str, np.ndarray], attacked: dict[str, np.ndarray], path: Path) -> None:
    stride = max(1, len(clean["t"]) // 250)
    ref = clean["refs"][::stride]
    clean_states = clean["states"][::stride]
    attacked_states = attacked["states"][::stride]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(clean["refs"][:, 0], clean["refs"][:, 1], "k--", lw=1.0, label="reference")
    clean_line, = ax.plot([], [], lw=2, label="IL clean")
    attacked_line, = ax.plot([], [], lw=2, label="IL + UAP")
    clean_dot, = ax.plot([], [], "o", ms=5)
    attacked_dot, = ax.plot([], [], "o", ms=5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Rollout Animation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    pad = 0.3
    xs = np.concatenate([ref[:, 0], clean_states[:, 0], attacked_states[:, 0]])
    ys = np.concatenate([ref[:, 1], clean_states[:, 1], attacked_states[:, 1]])
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal", adjustable="box")

    def update(frame: int):
        clean_line.set_data(clean_states[: frame + 1, 0], clean_states[: frame + 1, 1])
        attacked_line.set_data(attacked_states[: frame + 1, 0], attacked_states[: frame + 1, 1])
        clean_dot.set_data([clean_states[frame, 0]], [clean_states[frame, 1]])
        attacked_dot.set_data([attacked_states[frame, 0]], [attacked_states[frame, 1]])
        return clean_line, attacked_line, clean_dot, attacked_dot

    ani = animation.FuncAnimation(fig, update, frames=len(clean_states), interval=40, blit=True)
    ani.save(path, writer=animation.PillowWriter(fps=20))
    plt.close(fig)


def save_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_attack_perturbation_reports(results: list[dict], csv_path: Path, md_path: Path) -> None:
    rows = []
    for row in sorted(results, key=lambda item: (item["strategy"], item["epsilon"])):
        delta = row["delta"]
        delta_normalized = row.get("delta_normalized", [np.nan] * len(delta))
        mask = row["mask"]
        metadata = row.get("attack_metadata", {})
        out = {
            "strategy": row["strategy"],
            "attack_method": row.get("attack_method", metadata.get("attack_method", "")),
            "attack_score": metadata.get("score", np.nan),
            "attack_score_mode": metadata.get("score_mode", ""),
            "epsilon_normalized": row["epsilon"],
            "clean_rmse_pos": row["clean"]["rmse_pos"],
            "attacked_rmse_pos": row["attacked"]["rmse_pos"],
            "clean_rmse_theta": row["clean"]["rmse_theta"],
            "attacked_rmse_theta": row["attacked"]["rmse_theta"],
        }
        for idx, label in enumerate(FEATURE_LABELS[: len(delta)]):
            key = _field_label(label)
            out[f"mask_{key}"] = mask[idx]
            out[f"uap_raw_{key}"] = delta[idx]
            out[f"uap_normalized_{key}"] = delta_normalized[idx]
            feature_std = row.get("feature_std", [])
            if idx < len(feature_std):
                out[f"feature_std_{key}"] = feature_std[idx]
        rows.append(out)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# UAP Perturbation Report",
        "",
        "Epsilon is bounded in normalized feature units. The reported raw perturbation is added to the policy input before normalization.",
        "Raw perturbation equals normalized perturbation multiplied by the feature standard deviation. A large raw theta value usually means the unwrapped theta feature has a large dataset standard deviation, not that the normalized L-infinity projection failed.",
        "",
        "| Strategy | Method | Score | Epsilon norm | Raw UAP delta | Normalized UAP delta | Attacked pos RMSE | Attacked theta RMSE |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for result in sorted(results, key=lambda item: (item["strategy"], item["epsilon"])):
        delta = np.asarray(result["delta"], dtype=float)
        delta_normalized = np.asarray(result.get("delta_normalized", []), dtype=float)
        metadata = result.get("attack_metadata", {})
        method = result.get("attack_method", metadata.get("attack_method", ""))
        score = metadata.get("score", float("nan"))
        lines.append(
            f"| {result['strategy']} | {method} | {score:.6f} | {result['epsilon']:.3f} | {_format_delta(delta)} | "
            f"{_format_delta(delta_normalized)} | {result['attacked']['rmse_pos']:.6f} | "
            f"{result['attacked']['rmse_theta']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_attack_rollout_perturbation_reports(results: list[dict], csv_path: Path, md_path: Path) -> None:
    rows = []
    for row in sorted(results, key=lambda item: (item["strategy"], item["epsilon"])):
        delta = row["delta"]
        delta_normalized = row.get("delta_normalized", [np.nan] * len(delta))
        for rollout in row.get("per_rollout", []):
            clean = rollout["clean"]
            attacked = rollout["attacked"]
            metadata = row.get("attack_metadata", {})
            out = {
                "strategy": row["strategy"],
                "attack_method": row.get("attack_method", metadata.get("attack_method", "")),
                "attack_score": metadata.get("score", np.nan),
                "attack_score_mode": metadata.get("score_mode", ""),
                "epsilon_normalized": row["epsilon"],
                "rollout_id": rollout["rollout_id"],
                "trajectory_kind": rollout["trajectory_kind"],
                "clean_rmse_pos": clean["rmse_pos"],
                "attacked_rmse_pos": attacked["rmse_pos"],
                "clean_rmse_theta": clean["rmse_theta"],
                "attacked_rmse_theta": attacked["rmse_theta"],
            }
            for idx, label in enumerate(FEATURE_LABELS[: len(delta)]):
                key = _field_label(label)
                out[f"uap_raw_{key}"] = delta[idx]
                out[f"uap_normalized_{key}"] = delta_normalized[idx]
            rows.append(out)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Per-Rollout UAP Perturbation Report",
        "",
        "Each row repeats the actual universal perturbation applied during that rollout.",
        "Epsilon is bounded in normalized feature units. The raw perturbation is added before normalization.",
        "",
        "| Strategy | Method | Score | Epsilon norm | Rollout | Trajectory | Raw UAP delta | Attacked pos RMSE |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for result in sorted(results, key=lambda item: (item["strategy"], item["epsilon"])):
        delta = np.asarray(result["delta"], dtype=float)
        metadata = result.get("attack_metadata", {})
        method = result.get("attack_method", metadata.get("attack_method", ""))
        score = metadata.get("score", float("nan"))
        for rollout in result.get("per_rollout", []):
            attacked = rollout["attacked"]
            lines.append(
                f"| {result['strategy']} | {method} | {score:.6f} | {result['epsilon']:.3f} | {rollout['rollout_id']} | "
                f"{rollout['trajectory_kind']} | {_format_delta(delta)} | {attacked['rmse_pos']:.6f} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_gaussian_noise_report(results: list[dict], csv_path: Path, md_path: Path) -> None:
    rows = []
    for row in sorted(results, key=lambda item: (item["strategy"], item["std_normalized"])):
        rows.append(
            {
                "strategy": row["strategy"],
                "std_normalized": row["std_normalized"],
                "repeats": row["repeats"],
                "clean_rmse_pos": row["clean"]["rmse_pos"],
                "noisy_rmse_pos": row["attacked"]["rmse_pos"],
                "clean_rmse_theta": row["clean"]["rmse_theta"],
                "noisy_rmse_theta": row["attacked"]["rmse_theta"],
                "mean_abs_raw_noise": row["mean_abs_raw_noise"],
                "rms_raw_noise": row["rms_raw_noise"],
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Gaussian Noise Baseline",
        "",
        "Noise is i.i.d. per rollout step and bounded only statistically. Std is specified in normalized feature units.",
        "",
        "| Strategy | Std norm | Repeats | Noisy pos RMSE | Noisy theta RMSE | Mean abs raw noise | RMS raw noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {strategy} | {std_normalized:.3f} | {repeats} | {noisy_rmse_pos:.6f} | "
            "{noisy_rmse_theta:.6f} | {mean_abs_raw_noise:.6f} | {rms_raw_noise:.6f} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
