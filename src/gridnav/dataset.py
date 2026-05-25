"""
Data generation for the grid navigation experiment.
"""

import numpy as np

from gridnav.env import ACTION_NAMES
from gridnav.features import FEATURE_NAMES, extract_features
from gridnav.search import bfs_shortest_path, path_to_actions


ACTION_TO_ID = {action: index for index, action in enumerate(ACTION_NAMES)}
SPLIT_NAMES = np.array(["train", "val", "test"])


def generate_random_grid(height, width, obstacle_prob, rng):
    """Generate a random binary grid."""
    grid = rng.random((height, width)) < obstacle_prob
    return grid.astype(int)


def generate_corridor_grid(height, width, obstacle_prob, rng):
    """Generate a narrow-corridor style map."""
    grid = np.ones((height, width), dtype=int)
    row = int(rng.integers(height))
    col = int(rng.integers(width))

    target_free = int(height * width * (1 - obstacle_prob))
    grid[row, col] = 0
    free_count = 1

    # A random walk carves a connected set of cells, which gives maze-like maps.
    while free_count < target_free:
        action = ACTION_NAMES[int(rng.integers(len(ACTION_NAMES)))]
        if action == "up":
            row = max(0, row - 1)
        elif action == "down":
            row = min(height - 1, row + 1)
        elif action == "left":
            col = max(0, col - 1)
        else:
            col = min(width - 1, col + 1)

        if grid[row, col] == 1:
            grid[row, col] = 0
            free_count += 1

    return grid


def sample_free_position(grid, rng):
    """Sample one free cell from the grid."""
    free_positions = np.argwhere(grid == 0)
    index = rng.integers(len(free_positions))
    row, col = free_positions[index]

    return int(row), int(col)


def make_map_splits(num_maps, seed, train_ratio=0.7, val_ratio=0.1):
    """Assign each map to train, val, or test."""
    rng = np.random.default_rng(seed)
    ids = np.arange(num_maps)
    rng.shuffle(ids)

    n_train = int(num_maps * train_ratio)
    n_val = int(num_maps * val_ratio)

    split = np.full(num_maps, 2, dtype=int)
    split[ids[:n_train]] = 0
    split[ids[n_train:n_train + n_val]] = 1

    return split


def choose_map_size(height, width, size_options, rng):
    """Choose the map size for one generated map."""
    if size_options is None:
        return height, width

    index = int(rng.integers(len(size_options)))
    return size_options[index]


def pack_grids(grids):
    """Store grids without requiring every map to have the same shape."""
    arr = np.empty(len(grids), dtype=object)
    arr[:] = grids
    return arr


def generate_dataset(
    name,
    n_maps,
    obstacle_range,
    seed,
    height=None,
    width=None,
    size_options=None,
    corridor_ratio=0.0,
):
    """Generate one complete dataset."""
    rng = np.random.default_rng(seed)

    X = []
    y = []
    map_ids = []
    states = []

    grids = []
    starts = []
    goals = []
    shortest_lengths = []

    attempts = 0
    max_attempts = n_maps * 30

    # Some random maps are unreachable, so unreachable ones are skipped.
    while len(grids) < n_maps and attempts < max_attempts:
        attempts += 1

        obstacle_prob = rng.uniform(obstacle_range[0], obstacle_range[1])
        map_height, map_width = choose_map_size(height, width, size_options, rng)

        if rng.random() < corridor_ratio:
            grid = generate_corridor_grid(map_height, map_width, obstacle_prob, rng)
        else:
            grid = generate_random_grid(map_height, map_width, obstacle_prob, rng)

        if np.sum(grid == 0) < 2:
            continue

        start = sample_free_position(grid, rng)
        goal = sample_free_position(grid, rng)

        if start == goal:
            continue

        path = bfs_shortest_path(grid, start, goal)

        if path is None or len(path) < 2:
            continue

        actions = path_to_actions(path)
        map_id = len(grids)

        # Each step on the expert path becomes one supervised sample.
        for current, action in zip(path[:-1], actions):
            X.append(extract_features(grid, current, goal))
            y.append(ACTION_TO_ID[action])
            map_ids.append(map_id)
            states.append(current)

        grids.append(grid)
        starts.append(start)
        goals.append(goal)
        shortest_lengths.append(len(path) - 1)

    map_ids = np.array(map_ids, dtype=int)
    map_split = make_map_splits(len(grids), seed + 1)

    return {
        "name": np.array(name),
        "X": np.array(X, dtype=float),
        "y": np.array(y, dtype=int),
        "map_ids": map_ids,
        "states": np.array(states, dtype=int),
        "grids": pack_grids(grids),
        "starts": np.array(starts, dtype=int),
        "goals": np.array(goals, dtype=int),
        "shortest_lengths": np.array(shortest_lengths, dtype=int),
        "split": map_split[map_ids],
        "map_split": map_split,
        "feature_names": np.array(FEATURE_NAMES),
        "action_names": np.array(ACTION_NAMES),
        "split_names": SPLIT_NAMES,
    }
