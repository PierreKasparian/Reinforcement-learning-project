import enum
from dataclasses import dataclass
from typing import Optional


class Direction(enum.IntEnum):
    NORTH = 0  # (-1, 0)
    EAST = 1   # (0, 1)
    SOUTH = 2  # (1, 0)
    WEST = 3   # (0, -1)
    NONE = 4   # Calm / No wind


DIRECTION_OFFSETS = {
    Direction.NORTH: (-1, 0),
    Direction.EAST: (0, 1),
    Direction.SOUTH: (1, 0),
    Direction.WEST: (0, -1),
    Direction.NONE: (0, 0),
}


class ActionType(enum.IntEnum):
    NOOP = 0
    WATER_DROP = 1
    FIREBREAK = 2
    FIRE_TRUCK = 3


class Orientation(enum.IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


ACTION_COSTS = {
    ActionType.NOOP: 0,
    ActionType.WATER_DROP: 10,
    ActionType.FIREBREAK: 8,
    ActionType.FIRE_TRUCK: 3,
}


@dataclass
class ActionSpec:
    type: ActionType
    r: Optional[int] = None
    c: Optional[int] = None
    orientation: Optional[Orientation] = None


def n_actions(grid_size: int) -> int:
    """Total action count: 1 NOOP + 5 blocks of grid_size^2 targets.

    Blocks: water drop (H/V), firebreak (H/V) and fire truck (single cell).
    """
    return 1 + 5 * grid_size * grid_size


def decode_action(idx: int, grid_size: int) -> ActionSpec:
    """Convert a flat action index into an ActionSpec (inverse of encode_action)."""
    if not 0 <= idx < n_actions(grid_size):
        raise ValueError(f"action index {idx} out of range for grid size {grid_size}")

    if idx == 0:
        return ActionSpec(ActionType.NOOP)

    g2 = grid_size * grid_size
    k = idx - 1

    if k < 2 * g2:
        action_type = ActionType.WATER_DROP
    elif k < 4 * g2:
        action_type = ActionType.FIREBREAK
        k -= 2 * g2
    else:
        action_type = ActionType.FIRE_TRUCK
        k -= 4 * g2

    if action_type == ActionType.FIRE_TRUCK:
        r, c = divmod(k, grid_size)
        return ActionSpec(action_type, r, c)

    orientation = Orientation(k // g2)
    r, c = divmod(k % g2, grid_size)
    return ActionSpec(action_type, r, c, orientation)


def encode_action(spec: ActionSpec, grid_size: int) -> int:
    """Convert an ActionSpec into its flat action index (inverse of decode_action)."""
    if spec.type == ActionType.NOOP:
        return 0

    if spec.r is None or spec.c is None:
        raise ValueError("action requires a target cell")
    if not (0 <= spec.r < grid_size and 0 <= spec.c < grid_size):
        raise ValueError(f"cell ({spec.r}, {spec.c}) out of bounds for grid size {grid_size}")

    cell = spec.r * grid_size + spec.c
    g2 = grid_size * grid_size

    if spec.type == ActionType.WATER_DROP:
        if spec.orientation is None:
            raise ValueError("water drop requires an orientation")
        return 1 + spec.orientation * g2 + cell

    if spec.type == ActionType.FIREBREAK:
        if spec.orientation is None:
            raise ValueError("firebreak requires an orientation")
        return 1 + 2 * g2 + spec.orientation * g2 + cell

    return 1 + 4 * g2 + cell


def line_cells(spec: ActionSpec, grid_size: int, half_length: int = 2) -> list[tuple[int, int]]:
    """Return the target line cells (anchor +/- half_length), clipped to the grid."""
    if spec.r is None or spec.c is None:
        raise ValueError("action requires a target cell")
    if spec.orientation is None:
        raise ValueError("line action requires an orientation")

    cells = []
    if spec.orientation == Orientation.HORIZONTAL:
        for dc in range(-half_length, half_length + 1):
            c = spec.c + dc
            if 0 <= c < grid_size:
                cells.append((spec.r, c))
    else:
        for dr in range(-half_length, half_length + 1):
            r = spec.r + dr
            if 0 <= r < grid_size:
                cells.append((r, spec.c))
    return cells
