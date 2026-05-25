"""
Generate datasets and print a quick data check.
"""

from pathlib import Path
import argparse

import numpy as np

from gridnav.dataset import generate_dataset


DATASETS = {
    "dataset_a": {
        "n_maps": 1500,
        "height": 10,
        "width": 10,
        "obstacle_range": (0.10, 0.30),
        "seed": 42,
    },
    "dataset_b": {
        "n_maps": 900,
        "size_options": [(15, 15), (20, 20)],
        "obstacle_range": (0.20, 0.40),
        "corridor_ratio": 0.35,
        "seed": 100,
    },
}


def show_stats(data):
    """Print basic dataset information."""
    print(f"\n{data['name']}")
    print(f"maps: {len(data['grids'])}")
    print(f"samples: {len(data['X'])}")
    print(f"features: {data['X'].shape[1]}")
    print(f"feature names: {', '.join(data['feature_names'])}")

    for split_id, split_name in enumerate(data["split_names"]):
        idx = np.where(data["split"] == split_id)[0]
        map_count = np.sum(data["map_split"] == split_id)
        print(f"{split_name}: {len(idx)} samples, {map_count} maps")

    print("action counts:")
    for action_id, action in enumerate(data["action_names"]):
        print(f"  {action}: {np.sum(data['y'] == action_id)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="make smaller datasets for debugging")
    args = parser.parse_args()

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in DATASETS.items():
        cfg = cfg.copy()
        if args.quick:
            cfg["n_maps"] = 80 if name == "dataset_a" else 40

        data = generate_dataset(name=name, **cfg)
        np.savez_compressed(out_dir / f"{name}.npz", **data)

        print(f"saved: {out_dir / f'{name}.npz'}")
        show_stats(data)


if __name__ == "__main__":
    main()
