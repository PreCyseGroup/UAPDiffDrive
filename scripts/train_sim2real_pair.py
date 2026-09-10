#!/usr/bin/env python
"""Generate a published-method sim-to-real dataset and train nominal/FGSM residual policies."""

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

from train_adversarial_defense import adversarial_train, write_outputs as write_adversarial_outputs
from uap_il.dataset import generate_demonstrations, load_dataset, split_and_normalize
from uap_il.device import resolve_device
from uap_il.settings import load_settings
from uap_il.train import train_policy


CURVE_FOCUSED_FAMILIES = [
    "interpolated", "rounded_box", "circle", "ellipse", "slalom", "lemniscate",
    "interpolated", "sine_lane", "rounded_box", "circle", "lissajous", "trig_combo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--num-runs", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--adversarial-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=400000)
    parser.add_argument("--max-val-samples", type=int, default=80000)
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--skip-nominal", action="store_true", help="Reuse an already completed nominal checkpoint.")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/sim2real_domain_randomized_demonstrations.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/sim2real_retraining"))
    parser.add_argument("--nominal-checkpoint", type=Path, default=Path("artifacts/sim2real_nominal_policy.pt"))
    parser.add_argument("--nominal-norm", type=Path, default=Path("artifacts/sim2real_nominal_norm_stats.npz"))
    parser.add_argument("--adversarial-checkpoint", type=Path, default=Path("artifacts/sim2real_adversarial_policy.pt"))
    parser.add_argument("--adversarial-norm", type=Path, default=Path("artifacts/sim2real_adversarial_norm_stats.npz"))
    parser.add_argument("--nominal-init-checkpoint", type=Path, default=None)
    parser.add_argument("--adversarial-init-checkpoint", type=Path, default=None)
    parser.add_argument("--adversarial-stats-norm", type=Path, default=None)
    return parser.parse_args()


def save_nominal_outputs(out_dir: Path, history: dict[str, list[float]], metadata: dict[str, object]) -> None:
    nominal_dir = out_dir / "nominal_training"
    nominal_dir.mkdir(parents=True, exist_ok=True)
    with (nominal_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"]), 1):
            writer.writerow([epoch, train_loss, val_loss])
    (nominal_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set(xlabel="epoch", ylabel="normalized MSE", title="Sim-to-Real Nominal Training")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(nominal_dir / "training_curve.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for path in (args.dataset_path, args.nominal_checkpoint, args.nominal_norm,
                 args.adversarial_checkpoint, args.adversarial_norm):
        path.parent.mkdir(parents=True, exist_ok=True)
    settings = load_settings(args.config)
    device = resolve_device(args.device)
    args.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_dataset and args.dataset_path.exists():
        print(f"[data] loading {args.dataset_path}", flush=True)
        data = load_dataset(args.dataset_path)
    else:
        data = generate_demonstrations(
            args.dataset_path,
            args.num_runs,
            args.seed,
            settings.robot,
            settings.simulation,
            settings.deluca_gains,
            state_noise_xy_std=0.025,
            state_noise_theta_std=0.07,
            state_noise_clip_sigma=2.0,
            noisy_sample_fraction=0.25,
            dart_disturbed_rollout_fraction=0.40,
            dart_wheel_noise_std=0.22,
            dart_action_clip_sigma=2.0,
            initial_xy_min_distance=0.03,
            initial_xy_max_distance=0.32,
            initial_theta_error=0.22,
            dynamics_randomization_fraction=0.85,
            wheel_radius_scale_range=(0.94, 1.06),
            wheelbase_scale_range=(0.94, 1.06),
            wheel_gain_range=(0.88, 1.12),
            motor_time_constant_range=(0.02, 0.12),
            action_delay_steps_range=(0, 3),
            observation_delay_steps_range=(0, 2),
            trajectory_family_order=CURVE_FOCUSED_FAMILIES,
            target_mean_abs_wheel_range=(4.0, 7.2),
            min_reference_peak=5.4,
            max_reference_peak_fraction=0.92,
            log_progress=True,
        )

    stats = split_and_normalize(data["inputs"], data["targets"], args.seed, run_ids=data["run_ids"])
    dataset_metadata = json.loads(str(data["metadata_json"]))
    nominal_metadata_path = args.output_dir / "nominal_training" / "metadata.json"
    if args.skip_nominal:
        if not args.nominal_checkpoint.exists() or not args.nominal_norm.exists() or not nominal_metadata_path.exists():
            raise FileNotFoundError("--skip-nominal requires the nominal checkpoint, norm stats, and metadata.")
        nominal_metadata = json.loads(nominal_metadata_path.read_text(encoding="utf-8"))
        print(f"[train] reusing nominal checkpoint -> {args.nominal_checkpoint}", flush=True)
    else:
        print(f"[train] nominal residual policy on {len(stats['train_idx'])} available samples", flush=True)
        nominal_history = train_policy(
            data["inputs"], data["targets"], stats,
            args.nominal_checkpoint, args.nominal_norm,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate or settings.learning_rate,
            seed=args.seed, device=device,
            max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples,
            log_progress=True,
            initial_checkpoint_path=args.nominal_init_checkpoint,
        )
        nominal_metadata = {
            "method": "behavior_cloning_from_DART_and_dynamics_randomized_expert_rollouts",
            "dataset_path": str(args.dataset_path),
            "checkpoint_path": str(args.nominal_checkpoint),
            "norm_path": str(args.nominal_norm),
            "total_samples": int(len(data["inputs"])),
            "train_runs": int(len(stats["train_run_ids"])),
            "validation_runs": int(len(stats["val_run_ids"])),
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "completed_epochs": len(nominal_history["train_loss"]),
            "dataset_metadata": dataset_metadata,
        }
        save_nominal_outputs(args.output_dir, nominal_history, nominal_metadata)

    strategy = "ex_ey_theta" if "ex_ey_theta" in settings.attack_strategies else next(iter(settings.attack_strategies))
    defense_stats = dict(stats)
    if args.adversarial_stats_norm is not None:
        fixed_stats = np.load(args.adversarial_stats_norm)
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            defense_stats[key] = fixed_stats[key].astype(np.float32)
        print(f"[train] adversarial normalization -> {args.adversarial_stats_norm}", flush=True)
    defense_args = argparse.Namespace(
        learning_rate=args.learning_rate or settings.learning_rate,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        fgsm_epsilons=[0.10, 0.20, 0.35, 0.50],
        fgsm_fractions=[0.10, 0.15, 0.20, 0.25],
        fgsm_source_checkpoint=args.nominal_checkpoint,
        fgsm_source_norm=args.nominal_norm,
        fgsm_batch_size=args.batch_size,
        epochs=args.adversarial_epochs,
        adv_epsilon=0.35,
        random_epsilon=0.20,
        adv_loss_weight=0.75,
        random_loss_weight=0.25,
        checkpoint_path=args.adversarial_checkpoint,
        norm_path=args.adversarial_norm,
        output_dir=args.output_dir / "adversarial_training",
        initial_checkpoint=args.adversarial_init_checkpoint or args.nominal_checkpoint,
    )
    mask = np.asarray(settings.attack_strategies[strategy], dtype=np.float32)
    print("[train] adversarial residual policy", flush=True)
    defense_history = adversarial_train(data["inputs"], data["targets"], defense_stats, mask, defense_args, device, args.seed)
    defense_metadata = {
        "method": "DART_plus_dynamics_randomization_plus_multiscale_FGSM_adversarial_training",
        "dataset_path": str(args.dataset_path),
        "checkpoint_path": str(args.adversarial_checkpoint),
        "norm_path": str(args.adversarial_norm),
        "source_nominal_checkpoint": str(args.nominal_checkpoint),
        "strategy": strategy,
        "mask": mask.tolist(),
        "completed_epochs": len(defense_history["train_loss"]),
        "fixed_normalization_source": None if args.adversarial_stats_norm is None else str(args.adversarial_stats_norm),
    }
    write_adversarial_outputs(defense_args, defense_history, defense_metadata)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps({"nominal": nominal_metadata, "adversarial": defense_metadata}, indent=2), encoding="utf-8"
    )
    print(f"[done] dataset -> {args.dataset_path}", flush=True)
    print(f"[done] nominal -> {args.nominal_checkpoint}", flush=True)
    print(f"[done] adversarial -> {args.adversarial_checkpoint}", flush=True)


if __name__ == "__main__":
    main()
