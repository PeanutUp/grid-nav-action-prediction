"""
Basic grid-world environment utilities.

The grid uses:
- 0 for free cells
- 1 for obstacle cells

Positions are represented as (row, col).
Actions are represented by four strings: up, down, left, right.
"""

ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ACTION_NAMES = ["up", "down", "left", "right"]


def in_bounds(grid, pos):
    """Return True if the position is inside the grid."""
    row, col = pos
    height = grid.shape[0]
    width = grid.shape[1]

    return 0 <= row < height and 0 <= col < width


def is_free(grid, pos):
    """Return True if the position is not an obstacle."""
    row, col = pos

    return grid[row, col] == 0


def is_valid_pos(grid, pos):
    """Return True if the position is inside the grid and not blocked."""
    return in_bounds(grid, pos) and is_free(grid, pos)


def next_pos(pos, action):
    """Return the next position after applying an action."""
    dr, dc = ACTIONS[action]
    row, col = pos

    return row + dr, col + dc


def legal_actions(grid, pos):
    """Return all actions that can be legally taken from the current position."""
    actions = []

    # Try each action and keep it only if the next position is valid.
    for action in ACTION_NAMES:
        new_pos = next_pos(pos, action)

        if is_valid_pos(grid, new_pos):
            actions.append(action)

    return actions

