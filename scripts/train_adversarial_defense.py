#!/usr/bin/env python
"""Train a separate policy with state-noise and adversarial input training."""

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
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uap_il.config import Paths
from uap_il.dataset import generate_demonstrations, load_dataset, split_and_normalize
from uap_il.device import resolve_device
from uap_il.model import WheelMLP
from uap_il.settings import load_settings
from uap_il.train import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an adversarially robust defensive NN policy.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dataset-path", type=Path, default=Path("data/adversarial_defense_demonstrations.npz"))
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--num-runs", type=int, default=350)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=300000)
    parser.add_argument("--max-val-samples", type=int, default=60000)
    parser.add_argument("--state-noise-xy-std", type=float, default=0.035)
    parser.add_argument("--state-noise-theta-std", type=float, default=0.08)
    parser.add_argument("--state-noise-clip-sigma", type=float, default=2.5)
    parser.add_argument("--noisy-sample-fraction", type=float, default=0.30)
    parser.add_argument("--dart-disturbed-rollout-fraction", type=float, default=0.25)
    parser.add_argument("--dart-wheel-noise-std", type=float, default=0.15)
    parser.add_argument("--adv-epsilon", type=float, default=0.35)
    parser.add_argument("--random-epsilon", type=float, default=0.20)
    parser.add_argument("--adv-loss-weight", type=float, default=0.75)
    parser.add_argument("--random-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--fgsm-epsilons",
        type=float,
        nargs="+",
        default=[0.10, 0.20, 0.35, 0.50],
        help="Normalized FGSM magnitudes used to materialize extra training points.",
    )
    parser.add_argument(
        "--fgsm-fractions",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25],
        help="Fraction of the selected clean training split sampled at each FGSM magnitude.",
    )
    parser.add_argument("--fgsm-source-checkpoint", type=Path, default=Path("artifacts/il_policy.pt"))
    parser.add_argument("--fgsm-source-norm", type=Path, default=Path("artifacts/norm_stats.npz"))
    parser.add_argument("--fgsm-batch-size", type=int, default=8192)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("artifacts/adversarial_defense_policy.pt"))
    parser.add_argument("--norm-path", type=Path, default=Path("artifacts/adversarial_defense_norm_stats.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/adversarial_defense_training"))
    return parser.parse_args()


def _limited_indices(indices: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_count, replace=False)


def _project(delta: torch.Tensor, epsilon: float, mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(delta, -epsilon, epsilon) * mask


def _materialize_fgsm_points(
    inputs: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    stats: dict[str, np.ndarray],
    mask_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Create offline FGSM examples from the clean policy, with no validation leakage."""
    if len(args.fgsm_epsilons) != len(args.fgsm_fractions):
        raise ValueError("--fgsm-epsilons and --fgsm-fractions must have the same number of values.")
    if any(epsilon <= 0.0 for epsilon in args.fgsm_epsilons):
        raise ValueError("FGSM epsilons must be positive.")
    if any(not 0.0 <= fraction <= 1.0 for fraction in args.fgsm_fractions):
        raise ValueError("FGSM fractions must be in [0, 1].")
    if not args.fgsm_source_checkpoint.exists() or not args.fgsm_source_norm.exists():
        raise FileNotFoundError("The FGSM source checkpoint and normalization files are required.")

    source_model, source_stats = load_policy(args.fgsm_source_checkpoint, args.fgsm_source_norm, device=device)
    source_model.eval()
    source_x_mean = source_stats["x_mean"].astype(np.float32)
    source_x_std = source_stats["x_std"].astype(np.float32)
    source_y_mean = source_stats["y_mean"].astype(np.float32)
    source_y_std = source_stats["y_std"].astype(np.float32)
    mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    rng = np.random.default_rng(seed + 701)
    augmented_inputs: list[np.ndarray] = []
    augmented_targets: list[np.ndarray] = []
    counts: list[int] = []

    for distribution_id, (epsilon, fraction) in enumerate(zip(args.fgsm_epsilons, args.fgsm_fractions)):
        count = int(round(len(train_idx) * fraction))
        selected = rng.choice(train_idx, size=count, replace=False) if count else np.array([], dtype=np.int64)
        counts.append(count)
        if count == 0:
            continue
        raw = inputs[selected].astype(np.float32)
        target = targets[selected].astype(np.float32)
        source_x = torch.from_numpy((raw - source_x_mean) / source_x_std)
        source_y = torch.from_numpy((target - source_y_mean) / source_y_std)
        deltas = []
        for start in range(0, count, args.fgsm_batch_size):
            batch_x = source_x[start : start + args.fgsm_batch_size].to(device).requires_grad_(True)
            batch_y = source_y[start : start + args.fgsm_batch_size].to(device)
            loss = nn.functional.mse_loss(source_model(batch_x), batch_y)
            grad = torch.autograd.grad(loss, batch_x)[0]
            delta_norm = epsilon * torch.sign(grad).detach() * mask.view(1, -1)
            deltas.append(delta_norm.cpu().numpy())
        raw_delta = np.concatenate(deltas, axis=0) * source_x_std
        augmented_inputs.append(raw + raw_delta)
        augmented_targets.append(target)
        print(
            f"[fgsm-data] distribution={distribution_id + 1} epsilon={epsilon:.3f} "
            f"fraction={fraction:.3f} samples={count}",
            flush=True,
        )

    if augmented_inputs:
        train_inputs = np.concatenate([inputs[train_idx], *augmented_inputs], axis=0)
        train_targets = np.concatenate([targets[train_idx], *augmented_targets], axis=0)
    else:
        train_inputs = inputs[train_idx]
        train_targets = targets[train_idx]
    metadata = {
        "source_checkpoint": str(args.fgsm_source_checkpoint),
        "source_norm": str(args.fgsm_source_norm),
        "epsilons": [float(value) for value in args.fgsm_epsilons],
        "fractions_of_clean_training_split": [float(value) for value in args.fgsm_fractions],
        "counts": counts,
        "total_materialized": int(sum(counts)),
        "mask": mask_np.tolist(),
    }
    return train_inputs, train_targets, metadata


def adversarial_train(
    inputs: np.ndarray,
    targets: np.ndarray,
    stats: dict[str, np.ndarray],
    mask_np: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, list[float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = WheelMLP(
        input_dim=inputs.shape[1],
        x_mean=torch.from_numpy(stats["x_mean"].astype(np.float32)),
        x_std=torch.from_numpy(stats["x_std"].astype(np.float32)),
    ).to(device)
    initial_checkpoint = getattr(args, "initial_checkpoint", None)
    if initial_checkpoint is not None:
        initial_state = torch.load(initial_checkpoint, map_location=device, weights_only=True)
        initial_state = {
            key: value for key, value in initial_state.items()
            if key not in {"preprocess.x_mean", "preprocess.x_std"}
        }
        model.load_state_dict(initial_state, strict=False)
        print(f"[defense-train] initialized learnable weights from {initial_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_norm = ((inputs - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    y_norm = ((targets - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    x_tensor = torch.from_numpy(x_norm)
    y_tensor = torch.from_numpy(y_norm)

    train_idx = _limited_indices(np.asarray(stats["train_idx"], dtype=np.int64), args.max_train_samples, seed + 101)
    val_idx = _limited_indices(np.asarray(stats["val_idx"], dtype=np.int64), args.max_val_samples, seed + 107)
    fgsm_inputs, fgsm_targets, fgsm_metadata = _materialize_fgsm_points(
        inputs, targets, train_idx, stats, mask_np, args, device, seed
    )
    fgsm_x_norm = ((fgsm_inputs - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    fgsm_y_norm = ((fgsm_targets - stats["y_mean"]) / stats["y_std"]).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(fgsm_x_norm), torch.from_numpy(fgsm_y_norm))
    val_x = x_tensor[val_idx].to(device)
    val_y = y_tensor[val_idx].to(device)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)
    history = {"train_loss": [], "train_clean_loss": [], "train_adv_loss": [], "val_clean_loss": [], "val_adv_loss": []}
    best_val = float("inf")
    best_state = None
    stale = 0
    patience = 20

    print(
        f"[defense-train] samples={len(train_ds)} val={len(val_x)} epochs={args.epochs} "
        f"adv_eps={args.adv_epsilon:.3f} random_eps={args.random_epsilon:.3f} device={device}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_total = []
        epoch_clean = []
        epoch_adv = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            attack_x = batch_x.detach().clone().requires_grad_(True)
            attack_loss = loss_fn(model(attack_x), batch_y)
            grad = torch.autograd.grad(attack_loss, attack_x, retain_graph=False, create_graph=False)[0]
            adv_delta = _project(args.adv_epsilon * torch.sign(grad.detach()), args.adv_epsilon, mask)
            random_delta = _project(
                torch.empty_like(batch_x).uniform_(-args.random_epsilon, args.random_epsilon),
                args.random_epsilon,
                mask,
            )

            clean_loss = loss_fn(model(batch_x), batch_y)
            adv_loss = loss_fn(model(batch_x + adv_delta), batch_y)
            random_loss = loss_fn(model(batch_x + random_delta), batch_y)
            loss = clean_loss + args.adv_loss_weight * adv_loss + args.random_loss_weight * random_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_total.append(float(loss.detach().cpu().item()))
            epoch_clean.append(float(clean_loss.detach().cpu().item()))
            epoch_adv.append(float(adv_loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            clean_val = float(loss_fn(model(val_x), val_y).item())
            val_delta = args.adv_epsilon * mask.view(1, -1)
            adv_val = float(max(loss_fn(model(val_x + val_delta), val_y).item(), loss_fn(model(val_x - val_delta), val_y).item()))
        score = clean_val + args.adv_loss_weight * adv_val
        history["train_loss"].append(float(np.mean(epoch_total)))
        history["train_clean_loss"].append(float(np.mean(epoch_clean)))
        history["train_adv_loss"].append(float(np.mean(epoch_adv)))
        history["val_clean_loss"].append(clean_val)
        history["val_adv_loss"].append(adv_val)

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(
                f"[defense-train] epoch {epoch + 1:03d}/{args.epochs:03d} "
                f"train={history['train_loss'][-1]:.6f} val_clean={clean_val:.6f} val_adv={adv_val:.6f}",
                flush=True,
            )
        if score < best_val:
            best_val = score
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"[defense-train] early stop at epoch {epoch + 1}; stale={stale}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    args.norm_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, args.checkpoint_path)
    np.savez(
        args.norm_path,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
        train_run_ids=stats.get("train_run_ids", np.array([], dtype=np.int32)),
        val_run_ids=stats.get("val_run_ids", np.array([], dtype=np.int32)),
    )
    history["fgsm_data_metadata"] = fgsm_metadata
    return history


def write_outputs(args: argparse.Namespace, history: dict[str, list[float]], metadata: dict[str, object]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        metric_keys = [key for key, value in history.items() if isinstance(value, list)]
        writer.writerow(["epoch", *metric_keys])
        for epoch, values in enumerate(zip(*(history[key] for key in metric_keys)), start=1):
            writer.writerow([epoch, *values])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(history["train_clean_loss"], label="train clean")
    ax.plot(history["train_adv_loss"], label="train adversarial")
    ax.plot(history["val_clean_loss"], label="val clean")
    ax.plot(history["val_adv_loss"], label="val adversarial")
    ax.set_xlabel("epoch")
    ax.set_ylabel("normalized MSE")
    ax.set_title("Adversarial Defense Training")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    (args.output_dir / "fgsm_data_metadata.json").write_text(
        json.dumps(history["fgsm_data_metadata"], indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# Adversarial UAP Defense Training\n\n"
        "This run trains a separate NN policy checkpoint using two robustness mechanisms: "
        "state-noise recovery samples around the current robot pose during dataset generation, "
        "and materialized multi-distribution FGSM recovery samples plus on-the-fly adversarial perturbations "
        "in normalized policy-input space during training.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    seed = settings.seed if args.seed is None else args.seed
    args.batch_size = settings.batch_size if args.batch_size is None else args.batch_size
    args.learning_rate = settings.learning_rate if args.learning_rate is None else args.learning_rate
    device = resolve_device(args.device or settings.device)
    paths = Paths()
    paths.ensure()
    args.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_dataset and args.dataset_path.exists():
        data = load_dataset(args.dataset_path)
    else:
        data = generate_demonstrations(
            args.dataset_path,
            args.num_runs,
            seed,
            settings.robot,
            settings.simulation,
            settings.deluca_gains,
            state_noise_xy_std=args.state_noise_xy_std,
            state_noise_theta_std=args.state_noise_theta_std,
            state_noise_clip_sigma=args.state_noise_clip_sigma,
            noisy_sample_fraction=args.noisy_sample_fraction,
            dart_disturbed_rollout_fraction=args.dart_disturbed_rollout_fraction,
            dart_wheel_noise_std=args.dart_wheel_noise_std,
            log_progress=True,
        )
    stats = split_and_normalize(data["inputs"], data["targets"], seed, run_ids=data.get("run_ids"))
    strategy = args.strategy or ("ex_ey_theta" if "ex_ey_theta" in settings.attack_strategies else next(iter(settings.attack_strategies)))
    mask = np.asarray(settings.attack_strategies[strategy], dtype=np.float32)
    history = adversarial_train(data["inputs"], data["targets"], stats, mask, args, device, seed)
    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "norm_path": str(args.norm_path),
        "dataset_path": str(args.dataset_path),
        "num_runs": int(args.num_runs),
        "seed": int(seed),
        "strategy": strategy,
        "mask": mask.tolist(),
        "state_noise_xy_std": float(args.state_noise_xy_std),
        "state_noise_theta_std": float(args.state_noise_theta_std),
        "noisy_sample_fraction": float(args.noisy_sample_fraction),
        "dart_disturbed_rollout_fraction": float(args.dart_disturbed_rollout_fraction),
        "dart_wheel_noise_std": float(args.dart_wheel_noise_std),
        "adv_epsilon": float(args.adv_epsilon),
        "random_epsilon": float(args.random_epsilon),
        "adv_loss_weight": float(args.adv_loss_weight),
        "random_loss_weight": float(args.random_loss_weight),
        "completed_epochs": len(history["train_loss"]),
    }
    write_outputs(args, history, metadata)
    print(f"[done] defense checkpoint -> {args.checkpoint_path}", flush=True)
    print(f"[done] outputs -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
