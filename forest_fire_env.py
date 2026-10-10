import enum
from dataclasses import dataclass
from typing import Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from actions import (
    ACTION_COSTS,
    ActionSpec,
    ActionType,
    Direction,
    DIRECTION_OFFSETS,
    decode_action,
    line_cells,
    n_actions,
)


class CellState(enum.IntEnum):
    BURNT = 0
    HEALTHY = 1
    BURNING = 2
    FIREBREAK = 3
    WET = 4


@dataclass
class Wind:
    direction: Direction = Direction.NONE
    speed: float = 1.0          # Base wind strength when blowing
    is_active: bool = False     # Whether wind is currently present
    wind_prob: float = 0.6      # Target probability of wind presence overall

    def step(self, np_random: np.random.Generator, transition_prob: float = 0.15):
        """
        Transitions wind state:
        - Toggles between Calm (NONE) and Active Wind.
        - Shifts active wind direction by at most 90 degrees (adjacent direction).
        """
        if np_random.random() < transition_prob:
            if not self.is_active:
                # Wind starts blowing
                if np_random.random() < self.wind_prob:
                    self.is_active = True
                    self.direction = Direction(np_random.integers(0, 4))
            else:
                # Wind can stop, shift direction, or stay active
                roll = np_random.random()
                if roll < 0.3:
                    # Wind dies down
                    self.is_active = False
                    self.direction = Direction.NONE
                elif roll < 0.7:
                    # Shift 90 degrees clockwise or counterclockwise
                    turn = np_random.choice([-1, 1])
                    current_dir_val = self.direction.value if self.direction != Direction.NONE else 0
                    self.direction = Direction((current_dir_val + turn) % 4)

class Rain:
    def __init__(
        self,
        intensity: float = 0.5,     # Moisture units added per step during rain
        is_active: bool = True,
        rain_prob: float = 0.05,      # Chance of rain starting/stopping
    ):
        self.intensity = intensity
        self.is_active = is_active
        self.rain_prob = rain_prob

    def step(self, np_random: np.random.Generator, transition_prob: float = 0.1):
        """Randomly toggles rain state on/off."""
        if np_random.random() < transition_prob:
            if not self.is_active:
                if np_random.random() < self.rain_prob:
                    self.is_active = True
            else:
                if np_random.random() < 0.4:  # Chance rain stops
                    self.is_active = False

    def get_moisture_increase(self, r: int, c: int, grid_shape: tuple) -> float:
        """
        Returns moisture added to cell (r, c) this step.
        Uniform full-grid coverage for now.
        """
        if not self.is_active:
            return 0.0

        # Full-grid uniform coverage
        return self.intensity


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for the normalized reward components."""

    fire_damage: float = 50.0
    action_cost: float = 1.0
    time: float = 0.5
    success: float = 100.0
    failure: float = 50.0


class ForestFireEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        grid_size=15,
        p_spread=0.08,
        max_steps=150,
        max_fuel=100.0,
        max_moisture=100.0,
        base_burn_rate=2.0,
        burn_growth_rate=0.35,
        max_burn_rate=30.0,
        wind_speed=1.0,
        wind_factor=0.05,  # Scaling factor for wind influence on spread
        rain_intensity: float = 1.0,
        rain_prob: float = 0.1,
        max_water_drops: int = 12,
        max_firebreak_capacity: int = 15,
        max_budget: int = 150,
        action_costs=None,
        water_drop_amount: float = 50.0,
        truck_amount: float = 35.0,
        invalid_action_penalty: float = 0.5,
        line_half_length: int = 2,
        dry_rate: float = 0.5,
        reward_weights=None,
        **kwargs
    ):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.p_spread = p_spread
        self.max_steps = max_steps

        # Continuous state parameters
        self.max_fuel = max_fuel
        self.max_moisture = max_moisture
        self.base_burn_rate = base_burn_rate
        self.burn_growth_rate = burn_growth_rate
        self.max_burn_rate = max_burn_rate

        # Wind dynamics
        self.wind_factor = wind_factor
        self.default_wind_speed = wind_speed
        self.wind = Wind(direction=Direction.NORTH, speed=self.default_wind_speed)

        # Rain dynamics
        self.rain_intensity = rain_intensity
        self.rain_prob = rain_prob
        self.rain = Rain(intensity=rain_intensity, rain_prob=rain_prob, is_active=False)

        self.dry_rate = dry_rate
        self._watered_this_step = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Suppression resources
        self.max_water_drops = max_water_drops
        self.max_firebreak_capacity = max_firebreak_capacity
        self.max_budget = max_budget
        self.action_costs = dict(ACTION_COSTS if action_costs is None else action_costs)
        self.water_drop_amount = water_drop_amount
        self.truck_amount = truck_amount
        self.invalid_action_penalty = invalid_action_penalty
        self.line_half_length = line_half_length
        self.reward_weights = (
            reward_weights if reward_weights is not None else RewardWeights()
        )

        # State matrices
        self.grid = None
        self.fuel = None
        self.moisture = None
        self.burn_rate = None
        self.current_step = 0

        # Observation Space: 11 channels (Fuel, Moisture, Intensity, Firebreak, Wet,
        # Wind, Water Drops, Firebreak Capacity, Aircraft, Truck, Budget)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(11, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(n_actions(self.grid_size))

        self.renderer = None
        if self.render_mode is not None:
            from render import ForestFireRenderer
            self.renderer = ForestFireRenderer(
                grid_size=self.grid_size,
                render_fps=self.metadata["render_fps"],
                max_moisture=self.max_moisture,
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.grid = np.full((self.grid_size, self.grid_size), CellState.HEALTHY, dtype=np.int32)
        self.fuel = np.full((self.grid_size, self.grid_size), self.max_fuel, dtype=np.float32)
        self.initial_total_fuel = float(np.sum(self.fuel, dtype=np.float64))
        self.moisture = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.burn_rate = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._watered_this_step = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Initialize wind
        is_active = bool(self.np_random.random() < 0.6)  # 60% chance starting with wind
        initial_dir = Direction(self.np_random.integers(0, 4)) if is_active else Direction.NONE

        self.wind = Wind(
            direction=initial_dir,
            speed=self.default_wind_speed,
            is_active=is_active,
            wind_prob=0.6,
        )

        # Initialize rain
        self.rain = Rain(
            intensity=self.rain_intensity,
            rain_prob=self.rain_prob,
            is_active=False,
        )

        # Suppression resources
        self.water_drops_remaining = self.max_water_drops
        self.firebreak_capacity = self.max_firebreak_capacity
        self.aircraft_available = True # no restrictions implemented for the moment
        self.fire_truck_available = True
        self.budget_remaining = self.max_budget

        self.last_action = ActionSpec(ActionType.NOOP)
        self.action_valid = True
        self.failure_reason = None

        center = self.grid_size // 2
        self.grid[center, center] = CellState.BURNING
        self.burn_rate[center, center] = self.base_burn_rate

        return self._get_obs(), self._get_info()

    # ------- Environment -------

    def get_wind_vector_at(self, r: int, c: int) -> Tuple[float, float]:
        """Returns normalized (dr, dc) wind vector at cell (r, c)."""
        if not self.wind.is_active or self.wind.direction == Direction.NONE:
            return 0.0, 0.0
        return DIRECTION_OFFSETS[self.wind.direction]

    def _calculate_wind_modifier(self, source_r: int, source_c: int, target_r: int, target_c: int) -> float:
        """Calculates spread multiplier. Returns 1.0 when calm."""
        if not self.wind.is_active or self.wind.direction == Direction.NONE:
            return 1.0

        spread_dr = target_r - source_r
        spread_dc = target_c - source_c

        wind_dr, wind_dc = self.get_wind_vector_at(source_r, source_c)
        alignment = (spread_dr * wind_dr) + (spread_dc * wind_dc)

        modifier = 1.0 + (alignment * self.wind.speed * self.wind_factor)
        return max(0.00, modifier)

    def add_moisture(self, r, c, amount):
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return

        if self.grid[r, c] in [CellState.FIREBREAK, CellState.BURNT]:
            return

        self.moisture[r, c] = min(self.max_moisture, self.moisture[r, c] + amount)
        self._watered_this_step[r, c] = True

        if self.grid[r, c] == CellState.BURNING:
            suppression = amount * 0.1
            self.burn_rate[r, c] = max(0.5, self.burn_rate[r, c] - suppression)

        if self.moisture[r, c] >= self.max_moisture:
            self.grid[r, c] = CellState.WET
            self.burn_rate[r, c] = 0.0

    def _apply_drying(self):
        """Gradually reduces moisture towards 0 for cells not actively receiving water"""

        if self.dry_rate <= 0:
            return

        # Find cells that have moisture and were NOT watered during this step
        dry_mask = (self.moisture > 0.0) & (~self._watered_this_step)

        # Decrease moisture towards 0
        self.moisture[dry_mask] = np.maximum(0.0, self.moisture[dry_mask] - self.dry_rate)

        # Revert WET cells back to HEALTHY if moisture drops below max_moisture
        wet_and_drying = dry_mask & (self.grid == CellState.WET) & (self.moisture < self.max_moisture)
        self.grid[wet_and_drying] = CellState.HEALTHY

    def _update_environmental_factors(self):
        """Step environmental forces like Wind and Rain."""

        self._watered_this_step.fill(False)

        # Rain & apply moisture
        self.rain.step(self.np_random)

        if self.rain.is_active:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    added_moisture = self.rain.get_moisture_increase(r, c, (self.grid_size, self.grid_size))
                    if added_moisture > 0:
                        self.moisture[r, c] = min(
                            self.max_moisture,
                            self.moisture[r, c] + added_moisture
                        )

        # Wind
        self.wind.step(self.np_random, transition_prob=0.15)

    # ------- Actions -------

    def _action_cost(self, spec: ActionSpec) -> int:
        """Return the budget cost of the given action."""
        return self.action_costs[spec.type]

    def _is_action_valid(self, spec: ActionSpec) -> bool:
        """Check budget, resources and targets to decide if the action is legal."""
        if spec.type == ActionType.NOOP:
            return True

        if self._action_cost(spec) > self.budget_remaining:
            return False

        if spec.type == ActionType.WATER_DROP:
            if not (self.aircraft_available and self.water_drops_remaining > 0):
                return False
            cells = line_cells(spec, self.grid_size, self.line_half_length)
            return any(self.grid[r, c] in (CellState.HEALTHY, CellState.BURNING) for r, c in cells)

        if spec.type == ActionType.FIREBREAK:
            if self.firebreak_capacity <= 0:
                return False
            cells = line_cells(spec, self.grid_size, self.line_half_length)
            if any(self.grid[r, c] == CellState.BURNING for r, c in cells):
                return False
            return any(
                self.grid[r, c] in (CellState.HEALTHY, CellState.WET, CellState.BURNT)
                for r, c in cells
            )

        if spec.type == ActionType.FIRE_TRUCK:
            if not self.fire_truck_available:
                return False
            return self.grid[spec.r, spec.c] in (
                CellState.HEALTHY,
                CellState.BURNING,
                CellState.WET,
            )

        return False

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions for masked action selection."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        for idx in range(self.action_space.n):
            mask[idx] = self._is_action_valid(decode_action(idx, self.grid_size))
        return mask

    def _apply_water_drop(self, spec: ActionSpec):
        """Add water moisture along the target line."""
        for r, c in line_cells(spec, self.grid_size, self.line_half_length):
            self.add_moisture(r, c, self.water_drop_amount)

    def _apply_fire_truck(self, spec: ActionSpec):
        """Add fire truck moisture to its single target cell."""
        self.add_moisture(spec.r, spec.c, self.truck_amount)

    def _apply_firebreak(self, spec: ActionSpec):
        """Turn non-burning, non-firebreak cells along the target line into firebreaks."""
        for r, c in line_cells(spec, self.grid_size, self.line_half_length):
            if self.grid[r, c] in (CellState.HEALTHY, CellState.WET, CellState.BURNT):
                self.grid[r, c] = CellState.FIREBREAK
                self.fuel[r, c] = 0.0
                self.burn_rate[r, c] = 0.0

    def _get_obs(self):
        obs = np.zeros((11, self.grid_size, self.grid_size), dtype=np.float32)

        obs[0] = self.fuel / self.max_fuel
        obs[1] = self.moisture / self.max_moisture
        obs[2] = np.clip(self.burn_rate / self.max_burn_rate, 0.0, 1.0)
        obs[3] = (self.grid == CellState.FIREBREAK).astype(np.float32)
        obs[4] = (self.grid == CellState.WET).astype(np.float32)
        
        # Channel 5: Wind direction indicator (4.0 / 4.0 = 1.0 for CALM / NONE)
        obs[5] = self.wind.direction.value / 4.0

        # Channels 6-10: resource levels, broadcast over the grid
        obs[6] = self.water_drops_remaining / self.max_water_drops
        obs[7] = self.firebreak_capacity / self.max_firebreak_capacity
        obs[8] = float(self.aircraft_available)
        obs[9] = float(self.fire_truck_available)
        obs[10] = self.budget_remaining / self.max_budget

        return obs

    def _get_info(self):
        return {
            "healthy_count": int(np.sum(self.grid == CellState.HEALTHY)),
            "burning_count": int(np.sum(self.grid == CellState.BURNING)),
            "burnt_count": int(np.sum(self.grid == CellState.BURNT)),
            "wet_count": int(np.sum(self.grid == CellState.WET)),
            "firebreak_count": int(np.sum(self.grid == CellState.FIREBREAK)),
            "total_remaining_fuel": float(np.sum(self.fuel)),
            "total_moisture": float(np.sum(self.moisture)),
            "wind_direction": "CALM" if not self.wind.is_active else self.wind.direction.name,
            "wind_speed": self.wind.speed if self.wind.is_active else 0.0,
            "rain_active": self.rain.is_active,
            "rain_intensity": self.rain.intensity if self.rain.is_active else 0.0,
            "resources": {
                "water_drops_remaining": self.water_drops_remaining,
                "firebreak_capacity": self.firebreak_capacity,
                "aircraft_available": self.aircraft_available,
                "fire_truck_available": self.fire_truck_available,
                "budget_remaining": self.budget_remaining,
            },
            "action_valid": self.action_valid,
            "last_action": self.last_action,
            "failure_reason": self.failure_reason,
        }

    def _spread_fire(self):
        burning_coords = np.argwhere(self.grid == CellState.BURNING)
        new_fires = set()

        for r, c in burning_coords:
            # Moisture extinction
            if self.moisture[r, c] >= self.max_moisture:
                self.grid[r, c] = CellState.WET
                self.burn_rate[r, c] = 0.0
                continue
            elif self.moisture[r, c] > 0.0:
                p_extinguish = self.moisture[r, c] / (self.moisture[r, c] + self.burn_rate[r, c] * 10.0)
                if self.np_random.random() < p_extinguish:
                    self.grid[r, c] = CellState.HEALTHY
                    self.burn_rate[r, c] = 0.0
                    continue

                self.moisture[r, c] = max(0.0, self.moisture[r, c] - 2.0)

            # Fuel consumption
            self.fuel[r, c] -= self.burn_rate[r, c]

            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

            if self.fuel[r, c] <= 0.0:
                self.fuel[r, c] = 0.0
                self.grid[r, c] = CellState.BURNT
                self.burn_rate[r, c] = 0.0

                # Fuel exhaustion spread: Force ignite adjacent healthy trees
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid[nr, nc] == CellState.HEALTHY:
                            new_fires.add((nr, nc))
            else:
                curr_rate = self.burn_rate[r, c]
                logistic_delta = self.burn_growth_rate * curr_rate * (1.0 - (curr_rate / self.max_burn_rate))
                self.burn_rate[r, c] = min(self.max_burn_rate, curr_rate + logistic_delta)

                # Probabilistic spread with wind and moisture factors
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid[nr, nc] == CellState.HEALTHY:
                            moisture_factor = 1.0 - (self.moisture[nr, nc] / self.max_moisture)
                            wind_mod = self._calculate_wind_modifier(r, c, nr, nc)

                            effective_p = (
                                self.p_spread
                                * (1.0 + 0.05 * self.burn_rate[r, c])
                                * moisture_factor
                                * wind_mod
                            )

                            if self.np_random.random() < min(1.0, effective_p):
                                new_fires.add((nr, nc))

        # Apply new ignitions
        for r, c in new_fires:
            if self.grid[r, c] == CellState.HEALTHY and self.moisture[r, c] < self.max_moisture:
                self.grid[r, c] = CellState.BURNING
                self.burn_rate[r, c] = self.base_burn_rate

    def _calculate_reward(
        self,
        *,
        fuel_before_action,
        fuel_before_spread,
        budget_before,
        fires_before,
        active_fires,
        terminated,
        truncated,
    ):
        """Calculate the reward and expose its components without changing state."""
        w = self.reward_weights
        fuel_after = float(np.sum(self.fuel, dtype=np.float64))
        fuel_scale = max(self.initial_total_fuel, 1e-8)
        budget_scale = max(float(self.max_budget), 1.0)
        time_scale = max(int(self.max_steps), 1)

        # Only fuel lost during fire dynamics counts as fire damage.
        fuel_burned = max(0.0, fuel_before_spread - fuel_after)
        fuel_removed = max(0.0, fuel_before_action - fuel_before_spread)
        budget_spent = max(0.0, float(budget_before - self.budget_remaining))
        preserved_fraction = float(np.clip(fuel_after / fuel_scale, 0.0, 1.0))

        fires_extinguished = max(0, fires_before - active_fires)

        components = {
            "extinction": 1.5 * fires_extinguished,
            "fire_damage": -w.fire_damage * fuel_burned / fuel_scale,
            "action_cost": -w.action_cost * budget_spent / budget_scale,
            "time": -w.time / time_scale if active_fires > 0 else 0.0,
            "invalid_action": (
                -float(self.invalid_action_penalty)
                if not self.action_valid else 0.0
            ),
            "terminal": 0.0,
        }

        outcome = "ongoing"
        if terminated or truncated:
            success = (
                terminated
                and active_fires == 0
                and self.failure_reason is None
                and preserved_fraction > 0.0
            )
            if success:
                components["terminal"] = w.success * preserved_fraction
                outcome = "success"
            else:
                # A time limit reached without success is a failed episode.
                components["terminal"] = -w.failure
                outcome = self.failure_reason or (
                    "timeout" if truncated else "failure"
                )

        metrics = {
            "fuel_burned": fuel_burned,
            "fuel_removed": fuel_removed,
            "budget_spent": budget_spent,
            "preserved_fraction": preserved_fraction,
            "outcome": outcome,
        }
        return float(sum(components.values())), components, metrics

    def step(self, action):
        self.current_step += 1
        self._update_environmental_factors()
        spec = decode_action(int(action), self.grid_size)
        self.last_action = spec

        # Record resources before applying the intervention.
        budget_before = self.budget_remaining
        fuel_before_action = float(np.sum(self.fuel, dtype=np.float64))
        fires_before = int(np.sum(self.grid == CellState.BURNING))

        if not self._is_action_valid(spec):
            self.action_valid = False
        else:
            self.action_valid = True
            if spec.type == ActionType.WATER_DROP:
                self._apply_water_drop(spec)
                self.water_drops_remaining -= 1
            elif spec.type == ActionType.FIREBREAK:
                self._apply_firebreak(spec)
                self.firebreak_capacity -= 1
            elif spec.type == ActionType.FIRE_TRUCK:
                self._apply_fire_truck(spec)
            self.budget_remaining -= self._action_cost(spec)

        self._apply_drying()

        # Separate intervention-related fuel removal from fire damage.
        fuel_before_spread = float(np.sum(self.fuel, dtype=np.float64))
        self._spread_fire()

        active_fires = np.sum(self.grid == CellState.BURNING)
        healthy_trees = np.sum(self.grid == CellState.HEALTHY)
        budget_exhausted = self.budget_remaining == 0 and active_fires > 0

        # TODO: add failure termination when the burned area reaches the grid edge.
        terminated = active_fires == 0 or healthy_trees == 0 or budget_exhausted
        truncated = self.current_step >= self.max_steps

        self.failure_reason = None
        if terminated:
            if healthy_trees == 0:
                self.failure_reason = "no_healthy"
            elif budget_exhausted:
                self.failure_reason = "budget_exhausted"

        reward, components, metrics = self._calculate_reward(
            fuel_before_action=fuel_before_action,
            fuel_before_spread=fuel_before_spread,
            budget_before=budget_before,
            fires_before=fires_before,
            active_fires=active_fires,
            terminated=terminated,
            truncated=truncated,
        )
        info = self._get_info()
        info["reward_components"] = components
        info["reward_metrics"] = metrics

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.renderer is None:
                from render import ForestFireRenderer
                self.renderer = ForestFireRenderer(
                    grid_size=self.grid_size,
                    max_moisture=self.max_moisture,
                )

            info = self._get_info()

            # Format wind string
            if info["wind_direction"] in ["NONE", "CALM"] or info["wind_speed"] == 0.0:
                wind_str = "CALM (0.0)"
            else:
                wind_str = f"{info['wind_direction']} ({info['wind_speed']:.1f})"

            # Pass positional arguments in exact order expected by ForestFireRenderer:
            self.renderer.render(
                self.grid,
                self.burn_rate,
                self.moisture,
                wind_direction=info["wind_direction"],
                wind_speed=info["wind_speed"],
                wind_info=wind_str,
                rain_active=self.rain.is_active,
                rain_intensity=self.rain.intensity,
                resources=info["resources"],
                mode=self.render_mode,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
