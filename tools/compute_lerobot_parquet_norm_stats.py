"""Compute openpi norm stats directly from LeRobot parquet columns.

This is useful for large video datasets where the normal stats script would
decode images even though normalization only uses state and action arrays.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pyarrow.dataset as ds

import openpi.shared.normalize as normalize


def _read_column(table, name: str) -> np.ndarray:
    values = table[name].to_pylist()
    return np.asarray(values, dtype=np.float64)


def _stats(values: np.ndarray) -> normalize.NormStats:
    return normalize.NormStats(
        mean=values.mean(axis=0),
        std=values.std(axis=0),
        q01=np.quantile(values, 0.01, axis=0),
        q99=np.quantile(values, 0.99, axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--state-column", default="observation.state")
    parser.add_argument("--action-column", default="action")
    args = parser.parse_args()

    parquet_root = args.dataset_root / "data"
    dataset = ds.dataset(parquet_root, format="parquet")
    table = dataset.to_table(columns=[args.state_column, args.action_column])

    norm_stats = {
        "state": _stats(_read_column(table, args.state_column)),
        "actions": _stats(_read_column(table, args.action_column)),
    }
    normalize.save(args.output_dir, norm_stats)
    print(f"Wrote norm stats to {args.output_dir / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
