"""
BFS search and path-to-action conversion.
"""

from collections import deque

from gridnav.env import ACTION_NAMES, next_pos, is_valid_pos


def bfs_shortest_path(grid, start, goal):
    """
    Find a shortest path from start to goal using breadth-first search.

    Returns:
        A list of positions if the goal is reachable.
        None if no path exists.
    """
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        current = queue.popleft()

        # Once the goal is reached, reconstruct the path.
        if current == goal:
            return reconstruct_path(parent, goal)

        # Expand neighbors in a fixed action order.
        for action in ACTION_NAMES:
            neighbor = next_pos(current, action)

            if neighbor in visited:
                continue

            if not is_valid_pos(grid, neighbor):
                continue

            visited.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)

    return None


def reconstruct_path(parent, goal):
    """Reconstruct a path from the BFS parent dictionary."""
    path = []
    current = goal

    # Follow parent pointers backward from goal to start.
    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def path_to_actions(path):
    """Convert a position path into action labels."""
    actions = []

    # Compare each pair of consecutive positions to infer the action.
    for current, next_state in zip(path[:-1], path[1:]):
        dr = next_state[0] - current[0]
        dc = next_state[1] - current[1]

        if dr == -1 and dc == 0:
            actions.append("up")
        elif dr == 1 and dc == 0:
            actions.append("down")
        elif dr == 0 and dc == -1:
            actions.append("left")
        elif dr == 0 and dc == 1:
            actions.append("right")
        else:
            raise ValueError("Path contains non-adjacent positions.")

    return actions
