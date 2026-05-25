"""
Feature ablation for all learning models.
"""

from pathlib import Path
import csv
import os
import time

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score

from gridnav.features import FEATURE_NAMES
from gridnav.models import make_model_candidates


ABLATIONS = {
    "all": [],
    "no_relative_displacement": ["dx", "dy"],
    "no_distance": ["manhattan_distance", "euclidean_distance"],
    "no_goal_info": ["dx", "dy", "manhattan_distance", "euclidean_distance"],
    "no_blocked_flags": [
        "blocked_up",
        "blocked_down",
        "blocked_left",
        "blocked_right",
    ],
    "no_action_mask": [
        "legal_up",
        "legal_down",
        "legal_left",
        "legal_right",
    ],
    "no_local_obstacles": [
        "blocked_up",
        "blocked_down",
        "blocked_left",
        "blocked_right",
        "legal_up",
        "legal_down",
        "legal_left",
        "legal_right",
        "obstacles_3x3",
        "obstacles_5x5",
    ],
    "no_boundary": ["near_boundary"],
    "no_density": ["obstacle_density"],
}


def kept_columns(drop_names):
    """Return feature columns after removing a group."""
    drop = set(drop_names)
    return [i for i, name in enumerate(FEATURE_NAMES) if name not in drop]


def save_csv(rows, path):
    """Save ablation results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def split_indices(data, split_id):
    """Return sample indices for one split."""
    return np.where(data["split"] == split_id)[0]


def eval_row(ablation_name, model_name, dataset_name, params, n_features, y_true, y_pred,
             train_seconds, predict_seconds):
    """Build one result row."""
    return {
        "ablation": ablation_name,
        "model": model_name,
        "dataset": dataset_name,
        "best_params": params,
        "features": n_features,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }


def main():
    data_a = np.load(Path("data/processed/dataset_a.npz"), allow_pickle=True)
    data_b = np.load(Path("data/processed/dataset_b.npz"), allow_pickle=True)

    train_idx = split_indices(data_a, 0)
    val_idx = split_indices(data_a, 1)
    test_a_idx = split_indices(data_a, 2)
    test_b_idx = split_indices(data_b, 2)

    X_train = data_a["X"][train_idx]
    y_train = data_a["y"][train_idx]
    X_val = data_a["X"][val_idx]
    y_val = data_a["y"][val_idx]

    candidates = make_model_candidates(seed=0)
    rows = []
    tuning_rows = []

    for ablation_name, drop_names in ABLATIONS.items():
        cols = kept_columns(drop_names)
        print(f"\n{ablation_name}: {len(cols)} features")

        for model_name, model_candidates in candidates.items():
            best_model = None
            best_params = ""
            best_score = -1.0
            best_train_seconds = 0.0

            for params, candidate in model_candidates:
                model = clone(candidate)

                start = time.perf_counter()
                model.fit(X_train[:, cols], y_train)
                fit_seconds = time.perf_counter() - start

                val_pred = model.predict(X_val[:, cols])
                val_score = f1_score(y_val, val_pred, average="macro")
                tuning_rows.append({
                    "ablation": ablation_name,
                    "model": model_name,
                    "params": params,
                    "val_accuracy": accuracy_score(y_val, val_pred),
                    "val_macro_f1": val_score,
                    "fit_seconds": fit_seconds,
                })

                if val_score > best_score:
                    best_score = val_score
                    best_model = model
                    best_params = params
                    best_train_seconds = fit_seconds

            for dataset_name, data, idx in [
                ("A_test", data_a, test_a_idx),
                ("B_test", data_b, test_b_idx),
            ]:
                start = time.perf_counter()
                pred = best_model.predict(data["X"][idx][:, cols])
                predict_seconds = time.perf_counter() - start

                rows.append(eval_row(
                    ablation_name,
                    model_name,
                    dataset_name,
                    best_params,
                    len(cols),
                    data["y"][idx],
                    pred,
                    best_train_seconds,
                    predict_seconds,
                ))

            print(f"  {model_name}: {best_params}, val_macro_f1={best_score:.4f}")

    save_csv(rows, Path("outputs/metrics/ablation.csv"))
    save_csv(tuning_rows, Path("outputs/metrics/ablation_tuning.csv"))
    print("saved: outputs/metrics/ablation.csv")
    print("saved: outputs/metrics/ablation_tuning.csv")


if __name__ == "__main__":
    main()
