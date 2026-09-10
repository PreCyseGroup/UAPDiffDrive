#!/usr/bin/env python
"""Aggregate the established and dynamics-randomized demonstration datasets with disjoint run IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("data/demonstrations.npz"))
    parser.add_argument("--randomized", type=Path, default=Path("data/sim2real_domain_randomized_demonstrations.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/sim2real_aggregated_demonstrations.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = np.load(args.base, allow_pickle=False)
    randomized = np.load(args.randomized, allow_pickle=False)
    base_runs = base["run_ids"].astype(np.int64)
    randomized_runs = randomized["run_ids"].astype(np.int64)
    run_offset = int(np.max(base_runs)) + 1
    randomized_runs = randomized_runs + run_offset
    metadata = {
        "data_collection": "dataset_aggregation_of_established_and_DART_dynamics_randomized_demonstrations",
        "base_dataset": str(args.base),
        "randomized_dataset": str(args.randomized),
        "base_samples": int(len(base["inputs"])),
        "randomized_samples": int(len(randomized["inputs"])),
        "total_samples": int(len(base["inputs"]) + len(randomized["inputs"])),
        "base_runs": int(len(np.unique(base_runs))),
        "randomized_runs": int(len(np.unique(randomized_runs))),
        "run_id_offset": run_offset,
        "randomized_metadata": json.loads(str(randomized["metadata_json"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        inputs=np.concatenate([base["inputs"], randomized["inputs"]]),
        targets=np.concatenate([base["targets"], randomized["targets"]]),
        states=np.concatenate([base["states"], randomized["states"]]),
        refs=np.concatenate([base["refs"], randomized["refs"]]),
        run_ids=np.concatenate([base_runs, randomized_runs]),
        is_noisy_sample=np.concatenate([base["is_noisy_sample"], randomized["is_noisy_sample"]]),
        metadata_json=np.array(json.dumps(metadata, indent=2)),
    )
    print(f"[done] {metadata['total_samples']} samples, {metadata['base_runs'] + metadata['randomized_runs']} runs -> {args.output}")


if __name__ == "__main__":
    main()
