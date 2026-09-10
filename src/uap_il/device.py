from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown device '{requested}'. Use 'auto', 'cpu', or 'cuda'.")
    return torch.device(requested)
