#!/usr/bin/env python
"""Export the sim2real nominal training dataset as one documented CSV file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["ex", "ey", "theta", "xdr", "ydr", "xddr", "yddr", "vr"]
OUTPUT_COLUMNS = ["wheel_right", "wheel_left"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sim2real_aggregated_demonstrations.npz"),
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/demonstrations.npz"),
    )
    parser.add_argument(
        "--randomized-dataset",
        type=Path,
        default=Path("data/sim2real_domain_randomized_demonstrations.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sim2real_nominal_dataset"),
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as dataset:
        return json.loads(str(dataset["metadata_json"]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "sim2real_nominal_inputs_outputs.csv"
    readme_path = args.output_dir / "README.md"
    manifest_path = args.output_dir / "manifest.json"

    aggregate = np.load(args.dataset, allow_pickle=False)
    aggregate_metadata = json.loads(str(aggregate["metadata_json"]))
    base_metadata = load_metadata(args.base_dataset)
    randomized_metadata = load_metadata(args.randomized_dataset)
    run_offset = int(aggregate_metadata["run_id_offset"])

    trajectory_types: dict[int, str] = {
        int(run["run_id"]): str(run["spec"]["kind"])
        for run in base_metadata["runs"]
    }
    trajectory_types.update(
        {
            run_offset + int(run["run_id"]): str(run["spec"]["kind"])
            for run in randomized_metadata["runs"]
        }
    )

    run_ids = aggregate["run_ids"].astype(np.int64)
    unique_run_ids = np.unique(run_ids)
    missing = [int(run_id) for run_id in unique_run_ids if int(run_id) not in trajectory_types]
    if missing:
        raise RuntimeError(f"Missing trajectory metadata for run IDs: {missing[:10]}")

    inputs = aggregate["inputs"]
    targets = aggregate["targets"]
    noisy_flags = aggregate["is_noisy_sample"].astype(np.int8)
    row_count = len(inputs)
    if inputs.shape != (row_count, len(FEATURE_COLUMNS)):
        raise ValueError(f"Unexpected input shape: {inputs.shape}")
    if targets.shape != (row_count, len(OUTPUT_COLUMNS)):
        raise ValueError(f"Unexpected target shape: {targets.shape}")

    if csv_path.exists():
        csv_path.unlink()

    for start in range(0, row_count, args.chunk_size):
        end = min(start + args.chunk_size, row_count)
        chunk_run_ids = run_ids[start:end]
        frame = pd.DataFrame(inputs[start:end], columns=FEATURE_COLUMNS)
        frame.insert(0, "is_noisy_sample", noisy_flags[start:end])
        frame.insert(
            0,
            "data_source",
            np.where(chunk_run_ids < run_offset, "established", "domain_randomized"),
        )
        frame.insert(
            0,
            "trajectory_type",
            [trajectory_types[int(run_id)] for run_id in chunk_run_ids],
        )
        frame.insert(0, "trajectory_id", chunk_run_ids)
        frame.insert(0, "sample_id", np.arange(start, end, dtype=np.int64))
        frame[OUTPUT_COLUMNS] = targets[start:end]
        frame.to_csv(
            csv_path,
            mode="w" if start == 0 else "a",
            header=start == 0,
            index=False,
            float_format="%.9g",
        )
        print(f"[export] rows {start:,}..{end - 1:,}", flush=True)

    counts = {
        str(kind): int(sum(1 for run_id in unique_run_ids if trajectory_types[int(run_id)] == kind))
        for kind in sorted(set(trajectory_types.values()))
    }
    manifest = {
        "csv": csv_path.name,
        "source_dataset": str(args.dataset),
        "rows": int(row_count),
        "trajectories": int(len(unique_run_ids)),
        "columns": [
            "sample_id",
            "trajectory_id",
            "trajectory_type",
            "data_source",
            "is_noisy_sample",
            *FEATURE_COLUMNS,
            *OUTPUT_COLUMNS,
        ],
        "trajectory_counts_by_type": counts,
        "csv_size_bytes": int(csv_path.stat().st_size),
        "csv_sha256": sha256(csv_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme_path.write_text(
        f"""# Sim2Real Nominal Controller Dataset

`{csv_path.name}` is the raw input/output dataset used to train the
`artifacts/sim2real_nominal_policy.pt` residual wheel controller.

## Contents

- Rows: **{row_count:,}**
- Unique trajectories: **{len(unique_run_ids):,}**
- Source: `{args.dataset}`
- Manifest and checksum: `manifest.json`

Each row is one controller sample. `trajectory_id` is a globally unique rollout
ID, and `trajectory_type` identifies the reference family such as
`lemniscate`, `circle`, `interpolated`, or `rounded_box`.

## Input columns

| Column | Meaning | Unit |
|---|---|---|
| `ex` | Look-ahead position error along the robot body x-axis | m |
| `ey` | Look-ahead position error along the robot body y-axis | m |
| `theta` | Robot heading unwrapped to the reference-heading branch | rad |
| `xdr`, `ydr` | Reference world-frame velocity | m/s |
| `xddr`, `yddr` | Reference world-frame acceleration | m/s^2 |
| `vr` | Reference linear speed | m/s |

## Output columns

| Column | Meaning | Unit |
|---|---|---|
| `wheel_right` | Expert right-wheel angular-speed command | rad/s |
| `wheel_left` | Expert left-wheel angular-speed command | rad/s |

The CSV contains **raw, unnormalized** controller inputs and outputs.
`sim2real_nominal_norm_stats.npz` supplies the mean and standard deviation used
by the trained network.

`data_source` is `established` for the original demonstrations and
`domain_randomized` for the new DART/dynamics-randomized demonstrations.
`is_noisy_sample=1` marks a recovery/disturbed or locally augmented sample.
Training and validation should be split by `trajectory_id`, never by randomly
splitting individual rows, to avoid trajectory leakage.
""",
        encoding="utf-8",
    )
    print(f"[done] CSV -> {csv_path}", flush=True)
    print(f"[done] README -> {readme_path}", flush=True)
    print(f"[done] manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
