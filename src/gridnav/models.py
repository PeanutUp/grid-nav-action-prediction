"""
Models and simple baselines used in the experiment.
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin

from gridnav.env import ACTION_NAMES, is_valid_pos, next_pos


class KMeansMajorityClassifier(BaseEstimator, ClassifierMixin):
    """Cluster states first, then use the majority action in each cluster."""

    def __init__(self, n_clusters=16, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_ids = self.kmeans_.fit_predict(X)

        self.classes_ = np.unique(y)
        self.default_class_ = int(np.bincount(y).argmax())
        self.cluster_label_ = {}

        for cluster_id in range(self.n_clusters):
            labels = y[cluster_ids == cluster_id]
            if len(labels) == 0:
                self.cluster_label_[cluster_id] = self.default_class_
            else:
                self.cluster_label_[cluster_id] = int(np.bincount(labels).argmax())

        return self

    def predict(self, X):
        cluster_ids = self.kmeans_.predict(X)
        return np.array([self.cluster_label_[int(cid)] for cid in cluster_ids])


def make_models(seed=0):
    """Return one default model for each algorithm."""
    return {name: candidates[0][1] for name, candidates in make_model_candidates(seed).items()}


def make_model_candidates(seed=0):
    """Return validation candidates for each algorithm."""
    return {
        "knn": [
            (f"k={k},metric={metric}", Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=k, metric=metric)),
            ]))
            for k in [3, 5, 7, 11]
            for metric in ["euclidean", "manhattan"]
        ],
        "knn_weighted": [
            (f"k={k},metric={metric},distance", Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric)),
            ]))
            for k in [3, 5, 7, 11]
            for metric in ["euclidean", "manhattan"]
        ],
        "naive_bayes": [
            ("default", GaussianNB()),
        ],
        "logreg": [
            (f"C={c}", Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=1000, random_state=seed)),
            ]))
            for c in [0.1, 1.0, 10.0]
        ],
        "decision_tree": [
            (f"depth={depth}", DecisionTreeClassifier(max_depth=depth, random_state=seed))
            for depth in [6, 10, 14, None]
        ],
        "kmeans_majority": [
            (f"clusters={k}", Pipeline([
                ("scale", StandardScaler()),
                ("clf", KMeansMajorityClassifier(n_clusters=k, random_state=seed)),
            ]))
            for k in [8, 16, 32]
        ],
        "bagging_tree": [
            (f"trees={n}", BaggingClassifier(n_estimators=n, random_state=seed))
            for n in [30, 60, 100]
        ],
        "random_forest": [
            (f"trees={n},depth={depth}", RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                random_state=seed,
                n_jobs=1,
            ))
            for n in [80, 140]
            for depth in [10, 16, None]
        ],
        "adaboost": [
            (f"iters={n},lr={lr}", AdaBoostClassifier(
                n_estimators=n,
                learning_rate=lr,
                algorithm="SAMME",
                random_state=seed,
            ))
            for n in [50, 100]
            for lr in [0.5, 1.0]
        ],
        "stacking": [
            ("knn_lr_tree", StackingClassifier(
                estimators=[
                    ("knn", Pipeline([
                        ("scale", StandardScaler()),
                        ("clf", KNeighborsClassifier(n_neighbors=7)),
                    ])),
                    ("lr", Pipeline([
                        ("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                    ])),
                    ("tree", DecisionTreeClassifier(max_depth=10, random_state=seed)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
                cv=3,
                n_jobs=1,
            )),
            ("lr_tree_rf", StackingClassifier(
                estimators=[
                    ("lr", Pipeline([
                        ("scale", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
                    ])),
                    ("tree", DecisionTreeClassifier(max_depth=10, random_state=seed)),
                    ("rf", RandomForestClassifier(
                        n_estimators=80,
                        max_depth=12,
                        random_state=seed,
                        n_jobs=1,
                    )),
                ],
                final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
                cv=3,
                n_jobs=1,
            )),
        ],
    }


def greedy_action(grid, pos, goal):
    """Pick the legal action that gives the smallest Manhattan distance."""
    best_action = None
    best_distance = None

    for action in ACTION_NAMES:
        candidate = next_pos(pos, action)
        if not is_valid_pos(grid, candidate):
            continue

        distance = abs(goal[0] - candidate[0]) + abs(goal[1] - candidate[1])
        if best_distance is None or distance < best_distance:
            best_action = action
            best_distance = distance

    return best_action


def greedy_predict_samples(data, indices):
    """Predict labels for saved dataset samples using the greedy baseline."""
    action_to_id = {action: i for i, action in enumerate(ACTION_NAMES)}
    preds = []

    for i in indices:
        map_id = data["map_ids"][i]
        grid = data["grids"][map_id]
        goal = tuple(data["goals"][map_id])
        pos = tuple(data["states"][i])
        action = greedy_action(grid, pos, goal)
        preds.append(action_to_id[action])

    return np.array(preds, dtype=int)
