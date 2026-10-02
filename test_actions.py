"""Assert-based checks for the action system (no pytest dependency)."""

import numpy as np

from actions import (
    ACTION_COSTS,
    ActionSpec,
    ActionType,
    Orientation,
    decode_action,
    encode_action,
    line_cells,
    n_actions,
)
from forest_fire_env import CellState, ForestFireEnv


def make_env(grid_size=10, **kwargs):
    kwargs.setdefault("max_steps", 100)
    env = ForestFireEnv(render_mode=None, grid_size=grid_size, **kwargs)
    env.reset(seed=0)
    return env


def test_observation_channels():
    env = make_env()
    obs = env._get_obs()
    assert obs.shape == (11, 10, 10), obs.shape
    assert env.observation_space.contains(obs)
    assert np.allclose(obs[6], env.water_drops_remaining / env.max_water_drops)
    assert np.allclose(obs[7], env.firebreak_capacity / env.max_firebreak_capacity)
    assert np.allclose(obs[8], 1.0)
    assert np.allclose(obs[9], 1.0)
    assert np.allclose(obs[10], env.budget_remaining / env.max_budget)

    env.budget_remaining = 50
    assert np.allclose(env._get_obs()[10], 0.5)


def test_action_masks_shape():
    env = make_env()
    mask = env.action_masks()
    assert mask.shape == (n_actions(env.grid_size),)
    assert mask.dtype == bool
    assert mask[0]


def test_noop():
    env = make_env(p_spread=0.0)
    grid_before = env.grid.copy()
    budget_before = env.budget_remaining
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert np.array_equal(env.grid, grid_before)
    assert env.budget_remaining == budget_before
    assert info["action_valid"] is True
    assert info["last_action"].type == ActionType.NOOP


def test_rain_reset():
    env = make_env()
    env.rain.is_active = True
    env.reset(seed=0)
    assert env.rain.is_active is False


def test_water_drop_on_fire():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    env.grid[center, center] = CellState.BURNING
    env.burn_rate[center, center] = env.base_burn_rate

    spec = ActionSpec(ActionType.WATER_DROP, center, center, Orientation.HORIZONTAL)
    env._apply_water_drop(spec)

    assert env.burn_rate[center, center] < env.base_burn_rate
    for r, c in line_cells(spec, env.grid_size):
        assert env.moisture[r, c] == env.water_drop_amount


def test_water_drop_step_consumption():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    spec = ActionSpec(ActionType.WATER_DROP, center, center, Orientation.VERTICAL)
    idx = encode_action(spec, env.grid_size)
    assert env.action_masks()[idx]

    drops_before = env.water_drops_remaining
    budget_before = env.budget_remaining
    obs, reward, terminated, truncated, info = env.step(idx)

    assert reward == 0.0
    assert env.water_drops_remaining == drops_before - 1
    assert env.budget_remaining == budget_before - ACTION_COSTS[ActionType.WATER_DROP]
    for r, c in line_cells(spec, env.grid_size):
        assert env.moisture[r, c] > 0.0


def test_water_drop_without_drops():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    env.water_drops_remaining = 0
    idx = encode_action(
        ActionSpec(ActionType.WATER_DROP, center, center, Orientation.HORIZONTAL),
        env.grid_size,
    )
    assert not env.action_masks()[idx]

    budget_before = env.budget_remaining
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward == -env.invalid_action_penalty
    assert info["action_valid"] is False
    assert env.water_drops_remaining == 0
    assert env.budget_remaining == budget_before
    assert env.moisture[center, center] == 0.0


def test_firebreak_on_healthy_line():
    env = make_env(p_spread=0.0)
    spec = ActionSpec(ActionType.FIREBREAK, 2, 2, Orientation.HORIZONTAL)
    idx = encode_action(spec, env.grid_size)
    assert env.action_masks()[idx]

    capacity_before = env.firebreak_capacity
    budget_before = env.budget_remaining
    env.step(idx)

    for r, c in line_cells(spec, env.grid_size):
        assert env.grid[r, c] == CellState.FIREBREAK
        assert env.fuel[r, c] == 0.0
        assert env.burn_rate[r, c] == 0.0
    assert env.firebreak_capacity == capacity_before - 1
    assert env.budget_remaining == budget_before - ACTION_COSTS[ActionType.FIREBREAK]


def test_firebreak_with_burning_cell():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    idx = encode_action(
        ActionSpec(ActionType.FIREBREAK, center, center, Orientation.HORIZONTAL),
        env.grid_size,
    )
    assert not env.action_masks()[idx]

    capacity_before = env.firebreak_capacity
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward == -env.invalid_action_penalty
    assert env.firebreak_capacity == capacity_before
    assert env.grid[center, center] == CellState.BURNING


def test_firebreak_on_wet_and_burnt():
    env = make_env(p_spread=0.0)
    env.grid[2, 1] = CellState.WET
    env.grid[2, 2] = CellState.BURNT
    spec = ActionSpec(ActionType.FIREBREAK, 2, 2, Orientation.HORIZONTAL)
    idx = encode_action(spec, env.grid_size)
    assert env.action_masks()[idx]

    capacity_before = env.firebreak_capacity
    env.step(idx)
    for c in (1, 2):
        assert env.grid[2, c] == CellState.FIREBREAK
        assert env.fuel[2, c] == 0.0
        assert env.burn_rate[2, c] == 0.0
    assert env.firebreak_capacity == capacity_before - 1


def test_firebreak_on_already_firebreak_line():
    env = make_env(p_spread=0.0)
    spec = ActionSpec(ActionType.FIREBREAK, 2, 2, Orientation.HORIZONTAL)
    idx = encode_action(spec, env.grid_size)
    for r, c in line_cells(spec, env.grid_size):
        env.grid[r, c] = CellState.FIREBREAK
    assert not env.action_masks()[idx]


def test_fire_truck():
    env = make_env(p_spread=0.0)
    idx = encode_action(ActionSpec(ActionType.FIRE_TRUCK, 2, 2), env.grid_size)
    assert env.action_masks()[idx]

    budget_before = env.budget_remaining
    env.step(idx)
    assert env.moisture[2, 2] == env.truck_amount
    assert env.budget_remaining == budget_before - ACTION_COSTS[ActionType.FIRE_TRUCK]

    env.grid[2, 2] = CellState.BURNT
    assert not env.action_masks()[idx]


def test_fire_truck_on_burning_cell():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    idx = encode_action(ActionSpec(ActionType.FIRE_TRUCK, center, center), env.grid_size)
    assert env.action_masks()[idx]

    budget_before = env.budget_remaining
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward == 0.0
    assert info["action_valid"] is True
    assert env.budget_remaining == budget_before - ACTION_COSTS[ActionType.FIRE_TRUCK]


def test_fire_truck_reduces_burn_rate():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    spec = ActionSpec(ActionType.FIRE_TRUCK, center, center)
    env._apply_fire_truck(spec)
    assert env.burn_rate[center, center] < env.base_burn_rate
    assert env.moisture[center, center] == env.truck_amount


def test_fire_truck_on_wet_cell():
    env = make_env(p_spread=0.0)
    env.grid[2, 2] = CellState.WET
    idx = encode_action(ActionSpec(ActionType.FIRE_TRUCK, 2, 2), env.grid_size)
    assert env.action_masks()[idx]

    moisture_before = env.moisture[2, 2]
    env.step(idx)
    assert env.moisture[2, 2] == min(env.max_moisture, moisture_before + env.truck_amount)


def test_unaffordable_action():
    env = make_env(p_spread=0.0)
    env.budget_remaining = ACTION_COSTS[ActionType.FIRE_TRUCK] - 1
    idx = encode_action(ActionSpec(ActionType.FIRE_TRUCK, 2, 2), env.grid_size)
    assert not env.action_masks()[idx]

    budget_before = env.budget_remaining
    obs, reward, terminated, truncated, info = env.step(idx)
    assert reward == -env.invalid_action_penalty
    assert env.budget_remaining == budget_before


def test_budget_exhausted_termination():
    env = make_env(p_spread=0.0)
    env.budget_remaining = 0
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert info["failure_reason"] == "budget_exhausted"


def test_last_action_extinguish_with_zero_budget():
    env = make_env(p_spread=0.0)
    center = env.grid_size // 2
    env.moisture[center, center] = env.max_moisture - 1.0
    env.budget_remaining = ACTION_COSTS[ActionType.WATER_DROP]
    idx = encode_action(
        ActionSpec(ActionType.WATER_DROP, center, center, Orientation.HORIZONTAL),
        env.grid_size,
    )
    assert env.action_masks()[idx]

    obs, reward, terminated, truncated, info = env.step(idx)
    assert env.budget_remaining == 0
    assert terminated
    assert info["failure_reason"] is None
    assert np.sum(env.grid == CellState.BURNING) == 0


def test_encode_decode_round_trip():
    for grid_size in (10, 15):
        for idx in range(n_actions(grid_size)):
            spec = decode_action(idx, grid_size)
            assert encode_action(spec, grid_size) == idx

        g2 = grid_size * grid_size
        assert decode_action(0, grid_size).type == ActionType.NOOP

        spec = decode_action(1, grid_size)
        assert spec.type == ActionType.WATER_DROP
        assert spec.orientation == Orientation.HORIZONTAL
        assert (spec.r, spec.c) == (0, 0)

        spec = decode_action(1 + g2, grid_size)
        assert spec.type == ActionType.WATER_DROP
        assert spec.orientation == Orientation.VERTICAL
        assert (spec.r, spec.c) == (0, 0)

        spec = decode_action(1 + 2 * g2, grid_size)
        assert spec.type == ActionType.FIREBREAK
        assert spec.orientation == Orientation.HORIZONTAL
        assert (spec.r, spec.c) == (0, 0)

        spec = decode_action(1 + 4 * g2, grid_size)
        assert spec.type == ActionType.FIRE_TRUCK
        assert (spec.r, spec.c) == (0, 0)

        spec = decode_action(n_actions(grid_size) - 1, grid_size)
        assert spec.type == ActionType.FIRE_TRUCK
        assert (spec.r, spec.c) == (grid_size - 1, grid_size - 1)


def test_line_clipping():
    spec = ActionSpec(ActionType.WATER_DROP, 0, 0, Orientation.HORIZONTAL)
    assert line_cells(spec, 10) == [(0, 0), (0, 1), (0, 2)]

    spec = ActionSpec(ActionType.WATER_DROP, 0, 0, Orientation.VERTICAL)
    assert line_cells(spec, 10) == [(0, 0), (1, 0), (2, 0)]

    spec = ActionSpec(ActionType.FIREBREAK, 5, 9, Orientation.HORIZONTAL)
    assert line_cells(spec, 10) == [(5, 7), (5, 8), (5, 9)]


def test_action_index_out_of_range():
    env = make_env()
    try:
        env.step(n_actions(env.grid_size))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range action index")


def test_fuzz_random_steps():
    env = make_env(grid_size=15, max_steps=1000)
    rng = np.random.default_rng(123)

    for _ in range(500):
        mask = env.action_masks()
        if rng.random() < 0.9:
            action = int(rng.choice(np.flatnonzero(mask)))
        else:
            action = int(rng.integers(env.action_space.n))

        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (11, 15, 15)
        assert env.observation_space.contains(obs)
        assert env.water_drops_remaining >= 0
        assert env.firebreak_capacity >= 0
        assert env.budget_remaining >= 0

        if terminated or truncated:
            env.reset(seed=int(rng.integers(1_000_000)))


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK  {test.__name__}")
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
