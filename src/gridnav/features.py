"""
Feature extraction for one grid navigation state.
"""

import numpy as np

from gridnav.env import ACTION_NAMES, in_bounds, is_valid_pos, next_pos


FEATURE_NAMES = [
    "dx",
    "dy",
    "manhattan_distance",
    "euclidean_distance",
    "blocked_up",
    "blocked_down",
    "blocked_left",
    "blocked_right",
    "obstacles_3x3",
    "obstacles_5x5",
    "near_boundary",
    "obstacle_density",
    "legal_up",
    "legal_down",
    "legal_left",
    "legal_right",
]


def count_obstacles_in_window(grid, pos, radius):
    """Count obstacle cells in a local square window."""
    row, col = pos
    count = 0

    # Only real obstacle cells are counted; outside cells are handled elsewhere.
    for r in range(row - radius, row + radius + 1):
        for c in range(col - radius, col + radius + 1):
            if in_bounds(grid, (r, c)) and grid[r, c] == 1:
                count += 1

    return count


def extract_features(grid, current, goal):
    """Convert one navigation state into a fixed-length feature vector."""
    row, col = current
    goal_row, goal_col = goal

    dx = goal_row - row
    dy = goal_col - col
    manhattan = abs(dx) + abs(dy)
    euclidean = np.sqrt(dx ** 2 + dy ** 2)

    blocked_flags = []
    legal_flags = []

    # Keep both blocked flags and legal flags because the proposal lists both.
    for action in ACTION_NAMES:
        candidate = next_pos(current, action)
        legal = is_valid_pos(grid, candidate)

        blocked_flags.append(0 if legal else 1)
        legal_flags.append(1 if legal else 0)

    obstacles_3x3 = count_obstacles_in_window(grid, current, radius=1)
    obstacles_5x5 = count_obstacles_in_window(grid, current, radius=2)

    height, width = grid.shape

    # A cell is near boundary if it is on the outermost row or column.
    near_boundary = int(row == 0 or row == height - 1 or col == 0 or col == width - 1)

    obstacle_density = float(np.mean(grid == 1))

    features = [
        dx,
        dy,
        manhattan,
        euclidean,
        *blocked_flags,
        obstacles_3x3,
        obstacles_5x5,
        near_boundary,
        obstacle_density,
        *legal_flags,
    ]

    return np.array(features, dtype=float)
