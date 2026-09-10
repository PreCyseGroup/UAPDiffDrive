from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import POLICY_INPUT_DIM
from .model import WheelMLP


def train_policy(
    inputs: np.ndarray,
    targets: np.ndarray,
    stats: dict[str, np.ndarray],
    checkpoint_path: Path,
    norm_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device | str = "cpu",
    max_train_samples: int = 0,
    max_val_samples: int = 0,
    log_progress: bool = False,
    initial_checkpoint_path: Path | None = None,
) -> dict[str, list[float]]:
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = WheelMLP(
        input_dim=inputs.shape[1],
        x_mean=torch.from_numpy(stats["x_mean"].astype(np.float32)),
        x_std=torch.from_numpy(stats["x_std"].astype(np.float32)),
    ).to(device)
    if initial_checkpoint_path is not None:
        initial_state = torch.load(initial_checkpoint_path, map_location=device, weights_only=True)
        initial_state = {
            key: value for key, value in initial_state.items()
            if key not in {"preprocess.x_mean", "preprocess.x_std"}
        }
        model.load_state_dict(initial_state, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_norm = (inputs - stats["x_mean"]) / stats["x_std"]
    y_norm = (targets - stats["y_mean"]) / stats["y_std"]
    x_tensor = torch.from_numpy(x_norm.astype(np.float32))
    y_tensor = torch.from_numpy(y_norm.astype(np.float32))

    generator = torch.Generator().manual_seed(seed)
    train_idx = np.asarray(stats["train_idx"], dtype=np.int64)
    val_idx = np.asarray(stats["val_idx"], dtype=np.int64)
    if max_train_samples > 0 and len(train_idx) > max_train_samples:
        rng = np.random.default_rng(seed + 31)
        train_idx = rng.choice(train_idx, size=max_train_samples, replace=False)
    if max_val_samples > 0 and len(val_idx) > max_val_samples:
        rng = np.random.default_rng(seed + 37)
        val_idx = rng.choice(val_idx, size=max_val_samples, replace=False)

    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    val_x = x_tensor[val_idx].to(device)
    val_y = y_tensor[val_idx].to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    if log_progress:
        print(
            f"[train] samples={len(train_ds)} val={len(val_x)} "
            f"batch_size={batch_size} epochs={epochs} device={device}",
            flush=True,
        )
        if initial_checkpoint_path is not None:
            print(f"[train] initialized learnable weights from {initial_checkpoint_path}", flush=True)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    patience = 25
    stale = 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        train_loss = float(np.mean(losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if log_progress and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            print(
                f"[train] epoch {epoch + 1:03d}/{epochs:03d} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} best_val={min(best_val, val_loss):.6f}",
                flush=True,
            )

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                if log_progress:
                    print(f"[train] early stop at epoch {epoch + 1}; stale={stale}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if log_progress:
        print(f"[train] Saving checkpoint -> {checkpoint_path}", flush=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_path)
    np.savez(
        norm_path,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
        train_run_ids=stats.get("train_run_ids", np.array([], dtype=np.int32)),
        val_run_ids=stats.get("val_run_ids", np.array([], dtype=np.int32)),
    )
    return history


def load_policy(
    checkpoint_path: Path,
    norm_path: Path,
    device: torch.device | str = "cpu",
) -> tuple[WheelMLP, dict[str, np.ndarray]]:
    device = torch.device(device)
    stats_npz = np.load(norm_path)
    stats = {key: stats_npz[key].astype(np.float32) for key in stats_npz.files}
    input_dim = len(stats["x_mean"])
    if input_dim != POLICY_INPUT_DIM:
        raise RuntimeError(
            f"Normalization stats at {norm_path} have {input_dim} input features, but the current policy expects "
            f"{POLICY_INPUT_DIM}. Retrain the policy without --skip-train after changing the feature vector."
        )
    model = WheelMLP(
        input_dim=input_dim,
        x_mean=torch.from_numpy(stats["x_mean"].astype(np.float32)),
        x_std=torch.from_numpy(stats["x_std"].astype(np.float32)),
    ).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load policy checkpoint {checkpoint_path}. "
            f"The checkpoint may not match the current input dimension ({len(stats['x_mean'])}). "
            "Retrain the policy without --skip-train after changing the feature vector."
        ) from exc
    model.eval()
    return model, stats
