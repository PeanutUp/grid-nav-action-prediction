"""
Closed-loop navigation evaluation.
"""

from pathlib import Path
import csv
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np

from gridnav.env import is_valid_pos, next_pos
from gridnav.features import extract_features
from gridnav.models import greedy_action


def load_data(name):
    """Load one generated dataset."""
    return np.load(Path("data/processed") / f"{name}.npz", allow_pickle=True)


def model_action(model, grid, pos, goal, action_names):
    """Predict one action with a trained classifier."""
    x = extract_features(grid, pos, goal).reshape(1, -1)
    action_id = int(model.predict(x)[0])
    return str(action_names[action_id])


def rollout(grid, start, goal, shortest_len, choose_action):
    """Run one closed-loop episode."""
    pos = tuple(start)
    goal = tuple(goal)
    max_steps = grid.shape[0] * grid.shape[1] * 4

    for step in range(max_steps):
        if pos == goal:
            ratio = step / shortest_len if shortest_len > 0 else 1.0
            return True, step, ratio, False, False

        action = choose_action(pos)
        if action is None:
            return False, step, np.nan, True, False

        nxt = next_pos(pos, action)
        if not is_valid_pos(grid, nxt):
            return False, step + 1, np.nan, True, False

        pos = nxt

    return False, max_steps, np.nan, False, True


def eval_on_dataset(name, model_name, model, data):
    """Evaluate one model on all test maps of a dataset."""
    rows = []
    test_maps = np.where(data["map_split"] == 2)[0]
    action_names = data["action_names"]

    for map_id in test_maps:
        grid = data["grids"][map_id]
        start = data["starts"][map_id]
        goal = data["goals"][map_id]
        shortest_len = data["shortest_lengths"][map_id]

        if model_name == "greedy":
            choose = lambda pos: greedy_action(grid, pos, tuple(goal))
        else:
            choose = lambda pos: model_action(model, grid, pos, tuple(goal), action_names)

        rows.append(rollout(grid, start, goal, shortest_len, choose))

    arr = np.array(rows, dtype=object)
    success = arr[:, 0].astype(bool)
    ratios = arr[:, 2].astype(float)
    path_ratio = np.nan if np.all(np.isnan(ratios)) else np.nanmean(ratios)

    return {
        "model": model_name,
        "dataset": name,
        "success_rate": float(np.mean(success)),
        "avg_steps": float(np.mean(arr[:, 1].astype(float))),
        "path_ratio": float(path_ratio),
        "collision_rate": float(np.mean(arr[:, 3].astype(bool))),
        "timeout_rate": float(np.mean(arr[:, 4].astype(bool))),
    }


def save_csv(rows, path):
    """Save rollout metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_a = load_data("dataset_a")
    data_b = load_data("dataset_b")

    models = {"greedy": None}
    for path in sorted(Path("outputs/models").glob("*.joblib")):
        models[path.stem] = joblib.load(path)

    rows = []
    for model_name, model in models.items():
        print(f"rollout {model_name}...")
        rows.append(eval_on_dataset("A_test", model_name, model, data_a))
        rows.append(eval_on_dataset("B_test", model_name, model, data_b))

    save_csv(rows, Path("outputs/metrics/rollout.csv"))
    print("saved: outputs/metrics/rollout.csv")


if __name__ == "__main__":
    main()
