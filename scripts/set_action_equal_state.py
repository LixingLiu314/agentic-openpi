"""Set action = observation.state for every parquet in a LeRobot dataset.

Usage:
    python scripts/set_action_equal_state.py --dataset_dir aloha_cube
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main(dataset_dir: str) -> None:
    data_dir = Path(dataset_dir) / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Processing {len(parquet_files)} episodes in {data_dir} ...")

    for path in parquet_files:
        df = pd.read_parquet(path)
        states = np.stack(df["observation.state"].values)
        actions = np.stack(df["action"].values)

        max_diff = np.abs(actions - states).max()
        df["action"] = df["observation.state"]
        df.to_parquet(path, index=False)
        print(f"  {path.name}: max|action-state| was {max_diff:.4f} → set to 0")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()
    main(args.dataset_dir)
