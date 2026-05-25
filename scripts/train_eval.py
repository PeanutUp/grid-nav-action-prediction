"""
Train models on dataset A and evaluate single-step action prediction.
"""

from pathlib import Path
import csv
import os
import time

Path("outputs/cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "outputs/cache/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "outputs/cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from gridnav.models import greedy_predict_samples, make_model_candidates


import matplotlib.pyplot as plt


def load_data(name):
    """Load one generated dataset."""
    return np.load(Path("data/processed") / f"{name}.npz", allow_pickle=True)


def split_indices(data, split_id):
    """Get sample indices for train/val/test."""
    return np.where(data["split"] == split_id)[0]


def add_metrics(rows, model_name, dataset_name, y_true, y_pred, labels, train_seconds=0.0,
                predict_seconds=0.0, model_size_kb=0.0, best_params=""):
    """Append one metric row."""
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        zero_division=0,
    )

    row = {
        "model": model_name,
        "dataset": dataset_name,
        "best_params": best_params,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "model_size_kb": model_size_kb,
    }

    for i, action in enumerate(labels):
        row[f"precision_{action}"] = precision[i]
        row[f"recall_{action}"] = recall[i]

    rows.append(row)


def add_tuning_row(rows, model_name, params, val_y, val_pred, fit_seconds):
    """Store one validation result during hyperparameter search."""
    rows.append({
        "model": model_name,
        "params": params,
        "val_accuracy": accuracy_score(val_y, val_pred),
        "val_macro_f1": f1_score(val_y, val_pred, average="macro"),
        "fit_seconds": fit_seconds,
    })


def save_csv(rows, path):
    """Save metric rows as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_confusion(y_true, y_pred, labels, path):
    """Save one confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    data_a = load_data("dataset_a")
    data_b = load_data("dataset_b")

    train_idx = split_indices(data_a, 0)
    val_idx = split_indices(data_a, 1)
    test_a_idx = split_indices(data_a, 2)
    test_b_idx = split_indices(data_b, 2)

    X_train = data_a["X"][train_idx]
    y_train = data_a["y"][train_idx]
    labels = data_a["action_names"]

    rows = []
    tuning_rows = []
    model_dir = Path("outputs/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Greedy is not trained, but it is useful as a simple baseline.
    for dataset_name, data, idx in [
        ("A_val", data_a, val_idx),
        ("A_test", data_a, test_a_idx),
        ("B_test", data_b, test_b_idx),
    ]:
        start = time.perf_counter()
        pred = greedy_predict_samples(data, idx)
        predict_seconds = time.perf_counter() - start
        add_metrics(rows, "greedy", dataset_name, data["y"][idx], pred, labels,
                    predict_seconds=predict_seconds)

        if dataset_name == "A_test":
            save_confusion(
                data["y"][idx],
                pred,
                labels,
                Path("outputs/figures") / "cm_greedy.png",
            )

    for name, candidates in make_model_candidates(seed=0).items():
        print(f"tuning {name}...")
        best_model = None
        best_params = ""
        best_score = -1.0
        best_train_seconds = 0.0

        for params, candidate in candidates:
            model = clone(candidate)
            start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            val_pred = model.predict(data_a["X"][val_idx])
            val_score = f1_score(data_a["y"][val_idx], val_pred, average="macro")
            add_tuning_row(tuning_rows, name, params, data_a["y"][val_idx], val_pred, fit_seconds)

            if val_score > best_score:
                best_score = val_score
                best_model = model
                best_params = params
                best_train_seconds = fit_seconds

        model_path = model_dir / f"{name}.joblib"
        joblib.dump(best_model, model_path)
        model_size_kb = model_path.stat().st_size / 1024

        for dataset_name, data, idx in [
            ("A_val", data_a, val_idx),
            ("A_test", data_a, test_a_idx),
            ("B_test", data_b, test_b_idx),
        ]:
            start = time.perf_counter()
            pred = best_model.predict(data["X"][idx])
            predict_seconds = time.perf_counter() - start
            add_metrics(
                rows,
                name,
                dataset_name,
                data["y"][idx],
                pred,
                labels,
                train_seconds=best_train_seconds,
                predict_seconds=predict_seconds,
                model_size_kb=model_size_kb,
                best_params=best_params,
            )

            if dataset_name == "A_test":
                save_confusion(
                    data["y"][idx],
                    pred,
                    labels,
                    Path("outputs/figures") / f"cm_{name}.png",
                )

    save_csv(rows, Path("outputs/metrics/single_step.csv"))
    save_csv(tuning_rows, Path("outputs/metrics/tuning.csv"))
    print("saved: outputs/metrics/single_step.csv")
    print("saved: outputs/metrics/tuning.csv")


if __name__ == "__main__":
    main()
