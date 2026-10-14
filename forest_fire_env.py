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

NEIGHBOR_OFFSETS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1)
]

class CellState(enum.IntEnum):
    BURNT = 0
    HEALTHY = 1
    BURNING = 2
    FIREBREAK = 3
    WET = 4


@dataclass
class Wind:
    @property
    def speed(self) -> float:
        return self.current_speed if self.is_active else 0.0

    @property
    def direction(self):
        if not self.is_active or self.current_speed == 0.0:
            return Direction.NONE
        deg = np.degrees(self.current_direction_angle) % 360
        if 45 <= deg < 135:
            return Direction.SOUTH
        elif 135 <= deg < 225:
            return Direction.WEST
        elif 225 <= deg < 315:
            return Direction.NORTH
        else:
            return Direction.EAST

    def __init__(self, grid_size: int, max_speed: float = 3.0):
        self.grid_size = grid_size
        self.max_speed = max_speed
        self.vector_field = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
        self.is_active = False
        self.current_direction_angle = 0.0
        self.current_speed = 1.0

    def step(self, np_random: np.random.Generator, transition_prob: float = 0.15):
        """Smoothly transitions wind direction, speed, and localized patches."""
        if np_random.random() < transition_prob:
            if not self.is_active and np_random.random() < 0.6:
                self.is_active = True
                self.current_speed = np_random.uniform(1.0, 3.0)
                self.patch_center = np.array([
                    np_random.uniform(0, self.grid_size),
                    np_random.uniform(0, self.grid_size)
                ], dtype=np.float32)
            elif self.is_active and np_random.random() < 0.3:
                self.is_active = False

        if not self.is_active:
            self.current_speed = max(0.0, self.current_speed - 0.2)
            if self.current_speed == 0.0:
                self.vector_field *= 0.90
                return

        if hasattr(self, 'patch_center'):
            self.patch_center += np_random.uniform(-0.5, 0.5, size=2)
            self.patch_center = np.clip(self.patch_center, 0, self.grid_size)
        else:
            self.patch_center = np.array([self.grid_size / 2.0, self.grid_size / 2.0], dtype=np.float32)

        self.current_direction_angle += np_random.uniform(-0.25, 0.25)
        self.current_speed = np.clip(self.current_speed + np_random.uniform(-0.2, 0.2), 0.5, self.max_speed)

        base_dx = np.cos(self.current_direction_angle) * self.current_speed
        base_dy = np.sin(self.current_direction_angle) * self.current_speed

        target_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        pocket_radius = self.grid_size * 0.4

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                dist = np.hypot(r - self.patch_center[0], c - self.patch_center[1])
                if dist < pocket_radius:
                    spatial_mod = 0.5 * (1.0 + np.cos(np.pi * dist / pocket_radius))
                else:
                    spatial_mod = 0.0

                target_field[r, c, 0] = base_dx * spatial_mod
                target_field[r, c, 1] = base_dy * spatial_mod

        self.vector_field = 0.8 * self.vector_field + 0.2 * target_field

    def get_vector_at(self, r: int, c: int) -> Tuple[float, float]:
        """Returns the localized (dx, dy) wind vector at cell (r, c)."""
        if not self.is_active and np.all(self.vector_field == 0):
            return 0.0, 0.0
        return float(self.vector_field[r, c, 0]), float(self.vector_field[r, c, 1])


class Rain:
    @property
    def intensity(self) -> float:
        return self.current_intensity if self.is_active else 0.0

    def __init__(
        self,
        grid_size: int,
        max_intensity: float = 5.0,
        rain_prob: float = 0.2,
    ):
        self.grid_size = grid_size
        self.max_intensity = max_intensity
        self.rain_prob = rain_prob
        self.is_active = False
        self.current_intensity = 0.0
        self.storm_center = np.array([grid_size / 2.0, grid_size / 2.0], dtype=np.float32)
        self.storm_radius = grid_size * 0.35
        self.intensity_field = np.zeros((grid_size, grid_size), dtype=np.float32)

    def step(self, np_random: np.random.Generator, wind_field=None, transition_prob: float = 0.1):
        """Smoothly transitions rain presence, intensity, and drifts with wind."""
        if np_random.random() < transition_prob:
            if not self.is_active and np_random.random() < self.rain_prob:
                self.is_active = True
                self.current_intensity = np_random.uniform(1.0, 3.0)
                self.storm_center = np.array([
                    np_random.uniform(0, self.grid_size),
                    np_random.uniform(0, self.grid_size)
                ], dtype=np.float32)
            elif self.is_active and np_random.random() < 0.4:
                self.is_active = False

        if not self.is_active:
            self.current_intensity = max(0.0, self.current_intensity - 0.3)
            if self.current_intensity == 0.0:
                self.intensity_field.fill(0.0)
                return

        intensity_delta = np_random.uniform(-0.4, 0.4)
        self.current_intensity = np.clip(self.current_intensity + intensity_delta, 0.5, self.max_intensity)

        if wind_field is not None:
            sc_r = int(np.clip(self.storm_center[0], 0, self.grid_size - 1))
            sc_c = int(np.clip(self.storm_center[1], 0, self.grid_size - 1))
            wind_dx, wind_dy = wind_field[sc_r, sc_c]
            self.storm_center[0] += wind_dy * 0.5
            self.storm_center[1] += wind_dx * 0.5

        self.storm_center += np_random.uniform(-0.3, 0.3, size=2)
        self.storm_center = np.clip(self.storm_center, -2.0, self.grid_size + 2.0)

        self.intensity_field.fill(0.0)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                dist = np.hypot(r - self.storm_center[0], c - self.storm_center[1])
                if dist < self.storm_radius:
                    spatial_falloff = 0.5 * (1.0 + np.cos(np.pi * dist / self.storm_radius))
                    self.intensity_field[r, c] = self.current_intensity * spatial_falloff

    def get_moisture_increase(self, r: int, c: int, grid_shape: tuple) -> float:
        """Returns localized moisture added to cell (r, c) this step."""
        return float(self.intensity_field[r, c] / 2.0)


@dataclass(frozen=True)
class RewardWeights:
    """Configurable weights for the normalized reward components."""

    fire_damage: float = 1.5
    action_cost: float = 0.01
    time: float = 0.005
    suppression: float = 0.03
    success: float = 1.0
    failure: float = 0.5


class ForestFireEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        grid_size=15,
        p_spread=0.032,
        max_steps=300,
        max_fuel=100.0,
        max_moisture=100.0,
        base_burn_rate=2.0,
        burn_growth_rate=0.20,
        max_burn_rate=25.0,
        wind_speed=1.0,
        wind_factor=0.05,
        rain_intensity=5.0,
        rain_prob=0.1,
        budget_replenish_rate=1.5,
        water_drop_cooldown=4,
        cooldown_counter=0,
        aircraft_turnaround_steps=3,
        num_fire_trucks=2,
        truck_turnaround_steps=2,
        max_water_drops=12,
        max_budget=150,
        action_costs=None,
        water_drop_amount=48.0,
        truck_amount=35.0,
        dry_rate=0.2,
        reward_weights=None,
        **kwargs
    ):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.p_spread = p_spread
        self.max_steps = grid_size * 20

        # Dynamic Scaling based on Grid Area
        area_ratio = (self.grid_size / 15.0) ** 2
        self.max_water_drops = int(max_water_drops * area_ratio)
        self.firebreak_progress = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.max_construction_steps = 2  # 2 steps to complete a firebreak line
        self.max_budget = int(max_budget * area_ratio)

        self.line_half_length = max(1, self.grid_size // 12)

        self.min_initial_fires = 4
        self.max_initial_fires = 5

        # Continuous state parameters
        self.max_fuel = max_fuel
        self.max_moisture = max_moisture
        self.base_burn_rate = base_burn_rate
        self.burn_growth_rate = burn_growth_rate
        self.max_burn_rate = max_burn_rate

        # Wind dynamics
        self.wind_factor = wind_factor
        self.default_wind_speed = wind_speed
        self.wind = Wind(grid_size=self.grid_size, max_speed=3.0)

        # Rain dynamics
        self.rain_intensity = rain_intensity
        self.rain_prob = rain_prob
        self.rain = Rain(grid_size=self.grid_size, max_intensity=rain_intensity, rain_prob=rain_prob)

        self.dry_rate = dry_rate
        self._watered_this_step = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Suppression resources
        self.active_project_spec = None
        self.action_costs = dict(ACTION_COSTS if action_costs is None else action_costs)
        self.water_drop_amount = water_drop_amount
        self.truck_amount = truck_amount
        self.budget_replenish_rate = budget_replenish_rate
        self.water_drop_cooldown = water_drop_cooldown
        self.cooldown_counter = cooldown_counter

        # Vehicle turnaround
        self.aircraft_turnaround_steps = max(1, int(aircraft_turnaround_steps))
        self.num_fire_trucks = max(1, int(num_fire_trucks))
        self.truck_turnaround_steps = max(1, int(truck_turnaround_steps))
        self.aircraft_cooldown = 0
        self.truck_timers = np.zeros(self.num_fire_trucks, dtype=np.int32)

        self.reward_weights = (
            reward_weights if reward_weights is not None else RewardWeights()
        )

        # State matrices
        self.grid = None
        self.fuel = None
        self.moisture = None
        self.burn_rate = None
        self.current_step = 0

        self.observation_space = spaces.Dict({
            "spatial": spaces.Box(low=0.0, high=1.0, shape=(5, self.grid_size, self.grid_size), dtype=np.float32),
            "global": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        })

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
        self.moisture = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.burn_rate = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._watered_this_step = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Initialize wind
        self.wind = Wind(grid_size=self.grid_size, max_speed=3.0)
        if self.np_random.random() < 0.6:
            self.wind.is_active = True
            self.wind.current_speed = self.np_random.uniform(0.5, 1.5)
            self.wind.current_direction_angle = self.np_random.uniform(0, 2 * np.pi)
        else:
            self.wind.is_active = False
            self.wind.current_speed = 0.0

        # Initialize rain (keep rain off during pre-burn so it doesn't extinguish the initial seed)
        self.rain = Rain(
            grid_size=self.grid_size,
            max_intensity=self.rain_intensity,
            rain_prob=self.rain_prob,
        )

        # Suppression resources
        self.water_drops_remaining = self.max_water_drops
        self.cooldown_counter = 0
        self.aircraft_cooldown = 0
        self.truck_timers = np.zeros(self.num_fire_trucks, dtype=np.int32)
        self._sync_vehicle_availability()
        self.budget_remaining = self.max_budget
        self.firebreak_progress = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.active_project_spec = None

        self.last_action = ActionSpec(ActionType.NOOP)
        self.action_valid = True
        self.failure_reason = None

        center = self.grid_size // 2
        spawn_radius = max(1, self.grid_size // 6)

        r = int(self.np_random.integers(center - spawn_radius, center + spawn_radius + 1))
        c = int(self.np_random.integers(center - spawn_radius, center + spawn_radius + 1))

        # Spawn a 3-cell L-shaped cluster (spans 2 rows and 2 columns).
        # No 1D horizontal/vertical WATER_DROP or single-cell FIRE_TRUCK can hit all 3 cells on Step 1.
        dr = int(self.np_random.choice([-1, 1]))
        dc = int(self.np_random.choice([-1, 1]))
        core_cells = [
            (r, c),
            (int(np.clip(r + dr, 0, self.grid_size - 1)), c),
            (r, int(np.clip(c + dc, 0, self.grid_size - 1))),
        ]
        for cr, cc in core_cells:
            self.grid[cr, cc] = CellState.BURNING
            self.burn_rate[cr, cc] = self.base_burn_rate

        self.initial_total_fuel = float(np.sum(self.fuel, dtype=np.float64))

        return self._get_obs(), self._get_info()

    # ------- Environment -------

    def get_wind_vector_at(self, r: int, c: int) -> Tuple[float, float]:
        if not self.wind.is_active or self.wind.direction == Direction.NONE:
            return 0.0, 0.0
        return DIRECTION_OFFSETS[self.wind.direction]

    def _calculate_wind_modifier(self, source_r: int, source_c: int, target_r: int, target_c: int) -> float:
        spread_dr = target_r - source_r
        spread_dc = target_c - source_c

        dist = np.hypot(spread_dr, spread_dc)
        if dist == 0:
            return 1.0

        norm_dr = spread_dr / dist
        norm_dc = spread_dc / dist

        wind_dx, wind_dy = self.wind.get_vector_at(source_r, source_c)
        wind_mag = np.hypot(wind_dx, wind_dy)

        if wind_mag == 0.0:
            return 1.0

        wind_dir_dx = wind_dx / wind_mag
        wind_dir_dy = wind_dy / wind_mag

        alignment = (norm_dr * wind_dir_dx) + (norm_dc * wind_dir_dy)
        distance_penalty = 1.0 / dist

        modifier = 1.0 + (alignment * self.wind.speed * self.wind_factor * distance_penalty)
        return max(0.00, modifier)

    def add_moisture(self, r, c, amount):
        if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return

        if self.grid[r, c] in [CellState.FIREBREAK, CellState.BURNT]:
            return

        self.moisture[r, c] = min(self.max_moisture, self.moisture[r, c] + amount)
        self._watered_this_step[r, c] = True

        if self.grid[r, c] == CellState.BURNING:
            suppression = amount * 0.18
            self.burn_rate[r, c] = max(1.0, self.burn_rate[r, c] - suppression)

        if self.moisture[r, c] >= self.max_moisture:
            self.grid[r, c] = CellState.WET
            self.burn_rate[r, c] = 0.0

    def _apply_drying(self):
        if self.dry_rate <= 0:
            return

        dry_mask = (self.moisture > 0.0) & (~self._watered_this_step)
        self.moisture[dry_mask] = np.maximum(0.0, self.moisture[dry_mask] - self.dry_rate)

        wet_and_drying = dry_mask & (self.grid == CellState.WET) & (self.moisture < self.max_moisture)
        self.grid[wet_and_drying] = CellState.HEALTHY

    def _update_environmental_factors(self):
        self._watered_this_step.fill(False)
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

        self.wind.step(self.np_random, transition_prob=0.15)

    # ------- Vehicles -------

    def _sync_vehicle_availability(self):
        self.aircraft_available = self.aircraft_cooldown == 0
        self.fire_truck_available = bool(np.any(self.truck_timers == 0))

    def _dispatch_aircraft(self):
        self.aircraft_cooldown = self.aircraft_turnaround_steps
        self._sync_vehicle_availability()

    def _dispatch_fire_truck(self):
        idle = np.flatnonzero(self.truck_timers == 0)
        self.truck_timers[idle[0]] = self.truck_turnaround_steps
        self._sync_vehicle_availability()

    def _tick_vehicle_timers(self):
        self.aircraft_cooldown = max(0, self.aircraft_cooldown - 1)
        self.truck_timers = np.maximum(0, self.truck_timers - 1)
        self._sync_vehicle_availability()

    # ------- Actions -------

    def _action_cost(self, spec: ActionSpec) -> int:
        if (
            spec.type == ActionType.FIREBREAK
            and self.active_project_spec is not None
            and (spec.r, spec.c, spec.orientation) == self.active_project_spec
        ):
            return 0
        return self.action_costs[spec.type]

    def _is_action_valid(self, spec: ActionSpec) -> bool:
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
            cells = line_cells(spec, self.grid_size, self.line_half_length)
            if any(self.grid[r, c] == CellState.BURNING for r, c in cells):
                return False
            return any(self.grid[r, c] in (CellState.HEALTHY, CellState.WET) for r, c in cells)

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
        mask = np.zeros(self.action_space.n, dtype=bool)
        for idx in range(self.action_space.n):
            mask[idx] = self._is_action_valid(decode_action(idx, self.grid_size))
        return mask

    def _apply_water_drop(self, spec: ActionSpec):
        for r, c in line_cells(spec, self.grid_size, self.line_half_length):
            self.add_moisture(r, c, self.water_drop_amount)

    def _apply_fire_truck(self, spec: ActionSpec):
        self.add_moisture(spec.r, spec.c, self.truck_amount)

    def _apply_firebreak(self, spec: ActionSpec):
        target_spec = (spec.r, spec.c, spec.orientation)
        # If switching to a different project before finishing the previous one, clear unfinished progress
        if self.active_project_spec is not None and self.active_project_spec != target_spec:
            unfinished = (self.firebreak_progress > 0) & (self.firebreak_progress < self.max_construction_steps)
            self.firebreak_progress[unfinished] = 0
            self.fuel[unfinished] = self.max_fuel

        cells = line_cells(spec, self.grid_size, self.line_half_length)
        self.active_project_spec = target_spec
        all_completed = True

        for r, c in cells:
            if self.grid[r, c] in (CellState.HEALTHY, CellState.WET):
                self.firebreak_progress[r, c] = min(
                    self.max_construction_steps,
                    self.firebreak_progress[r, c] + 1
                )
                cleared_fraction = self.firebreak_progress[r, c] / self.max_construction_steps
                self.fuel[r, c] = max(0.0, self.max_fuel * (1.0 - cleared_fraction))

                if self.firebreak_progress[r, c] >= self.max_construction_steps:
                    self.grid[r, c] = CellState.FIREBREAK
                    self.fuel[r, c] = 0.0
                    self.burn_rate[r, c] = 0.0
                else:
                    all_completed = False

        if all_completed:
            self.active_project_spec = None

    def _get_obs(self):
        spatial = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        spatial[0] = self.fuel / self.max_fuel
        spatial[1] = self.moisture / self.max_moisture
        spatial[2] = np.clip(self.burn_rate / self.max_burn_rate, 0.0, 1.0)
        spatial[3] = self.firebreak_progress.astype(np.float32) / float(self.max_construction_steps)
        spatial[4] = (self.grid == CellState.WET).astype(np.float32)

        global_obs = np.array([
            self.wind.direction.value / 4.0,
            self.water_drops_remaining / max(1, self.max_water_drops),
            1.0 if self.active_project_spec is not None else 0.0,
            1.0 - (self.aircraft_cooldown / self.aircraft_turnaround_steps),
            float(np.mean(self.truck_timers == 0)),
            self.budget_remaining / max(1, self.max_budget)
        ], dtype=np.float32)

        return {"spatial": spatial, "global": global_obs}

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
            "wind_speed": self.wind.speed,
            "wind_vector_field": self.wind.vector_field,
            "rain_active": self.rain.is_active,
            "rain_intensity": self.rain.intensity if self.rain.is_active else 0.0,
            "rain_intensity_field": self.rain.intensity_field,
            "resources": {
                "water_drops_remaining": self.water_drops_remaining,
                "aircraft_available": self.aircraft_available,
                "fire_truck_available": self.fire_truck_available,
                "aircraft_cooldown": int(self.aircraft_cooldown),
                "trucks_idle": int(np.sum(self.truck_timers == 0)),
                "num_fire_trucks": self.num_fire_trucks,
                "budget_remaining": self.budget_remaining,
            },
            "action_valid": self.action_valid,
            "last_action": self.last_action,
            "failure_reason": self.failure_reason,
        }

    def _spread_fire(self):
        burning_mask = (self.grid == CellState.BURNING)

        p_extinguish = np.zeros_like(self.moisture)
        moisture_active = burning_mask & (self.moisture > 0)
        p_extinguish[moisture_active] = self.moisture[moisture_active] / (
            self.moisture[moisture_active] + self.burn_rate[moisture_active] * 6.0
        )

        rolls = self.np_random.random(size=self.grid.shape)
        extinguished_mask = moisture_active & (rolls < p_extinguish)

        self.grid[extinguished_mask] = CellState.HEALTHY
        self.burn_rate[extinguished_mask] = 0.0

        still_burning_mask = burning_mask & ~extinguished_mask
        self.moisture[still_burning_mask] = np.maximum(0.0, self.moisture[still_burning_mask] - 2.0)

        self.fuel[still_burning_mask] -= self.burn_rate[still_burning_mask]

        curr_rate = self.burn_rate[still_burning_mask]
        logistic_delta = self.burn_growth_rate * curr_rate * (1.0 - (curr_rate / self.max_burn_rate))
        self.burn_rate[still_burning_mask] = np.minimum(self.max_burn_rate, curr_rate + logistic_delta)

        burnout_mask = still_burning_mask & (self.fuel <= 0.0)
        burnout_rates = self.burn_rate.copy()
        self.fuel[burnout_mask] = 0.0
        self.grid[burnout_mask] = CellState.BURNT
        self.burn_rate[burnout_mask] = 0.0

        new_fires = set()

        burnout_coords = np.argwhere(burnout_mask)
        for r, c in burnout_coords:
            intensity_ratio = burnout_rates[r, c] / self.max_burn_rate
            for dr, dc in NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr, nc] == CellState.HEALTHY:
                        moisture_factor = 1.0 - (self.moisture[nr, nc] / self.max_moisture)
                        wind_mod = self._calculate_wind_modifier(r, c, nr, nc)
                        dist = np.hypot(dr, dc)
                        base_prob = self.p_spread * (1.0 + intensity_ratio) * moisture_factor * wind_mod
                        fuel_fraction = self.fuel[nr, nc] / self.max_fuel
                        effective_p = base_prob * (1.0 / dist) * fuel_fraction
                        if self.np_random.random() < min(1.0, effective_p):
                            new_fires.add((nr, nc))

        active_coords = np.argwhere(still_burning_mask & ~burnout_mask)
        for r, c in active_coords:
            for dr, dc in NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr, nc] == CellState.HEALTHY:
                        moisture_factor = 1.0 - (self.moisture[nr, nc] / self.max_moisture)
                        wind_mod = self._calculate_wind_modifier(r, c, nr, nc)
                        dist = np.hypot(dr, dc)
                        base_probability = self.p_spread * (1.0 + 0.05 * self.burn_rate[r, c]) * moisture_factor * wind_mod
                        fuel_fraction = self.fuel[nr, nc] / self.max_fuel
                        effective_p = base_probability * (1.0 / dist) * fuel_fraction
                        if self.np_random.random() < min(1.0, effective_p):
                            new_fires.add((nr, nc))

        for r, c in new_fires:
            if (
                self.grid[r, c] == CellState.HEALTHY
                and not extinguished_mask[r, c]
                and not self._watered_this_step[r, c]
                and self.moisture[r, c] < self.max_moisture
            ):
                self.grid[r, c] = CellState.BURNING
                self.burn_rate[r, c] = self.base_burn_rate

        interrupted_mask = (
            (self.grid == CellState.BURNING)
            & (self.firebreak_progress > 0)
            & (self.firebreak_progress < self.max_construction_steps)
        )
        if np.any(interrupted_mask):
            self.firebreak_progress[interrupted_mask] = 0
            if self.active_project_spec is not None:
                proj_spec = ActionSpec(
                    ActionType.FIREBREAK,
                    self.active_project_spec[0],
                    self.active_project_spec[1],
                    self.active_project_spec[2],
                )
                proj_cells = line_cells(proj_spec, self.grid_size, self.line_half_length)
                if any(interrupted_mask[r, c] for r, c in proj_cells):
                    self.active_project_spec = None

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
        burn_rate_suppressed,
    ):
        w = self.reward_weights
        fuel_after = float(np.sum(self.fuel, dtype=np.float64))
        fuel_scale = max(self.initial_total_fuel, 1e-8)
        budget_scale = max(float(self.max_budget), 1.0)
        time_scale = max(int(self.max_steps), 1)

        fuel_burned = max(0.0, fuel_before_spread - fuel_after)
        fuel_removed = max(0.0, fuel_before_action - fuel_before_spread)
        budget_spent = max(0.0, float(budget_before - self.budget_remaining))
        preserved_fraction = float(np.clip(fuel_after / fuel_scale, 0.0, 1.0))

        components = {
            "fire_damage": -w.fire_damage * fuel_burned / fuel_scale,
            "action_cost": -w.action_cost * budget_spent / budget_scale,
            "suppression": w.suppression * (burn_rate_suppressed / max(float(self.max_burn_rate), 1.0)),
            "time": -w.time / time_scale if active_fires > 0 else 0.0,
            "terminal": 0.0,
        }

        outcome = "ongoing"
        if terminated or truncated:
            success = (
                terminated
                and active_fires == 0
                and self.failure_reason is None
                and preserved_fraction >= 0.40
            )
            if success:
                components["terminal"] = w.success * preserved_fraction
                outcome = "success"
            else:
                burned_fraction = 1.0 - preserved_fraction
                components["terminal"] = -w.failure * burned_fraction
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

        budget_before = self.budget_remaining
        fuel_before_action = float(np.sum(self.fuel, dtype=np.float64))
        fires_before = int(np.sum(self.grid == CellState.BURNING))
        burn_rate_before_action = float(np.sum(self.burn_rate, dtype=np.float64))

        self.budget_remaining = min(
            self.max_budget,
            self.budget_remaining + self.budget_replenish_rate
        )

        if self.water_drops_remaining < self.max_water_drops:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.water_drop_cooldown:
                self.water_drops_remaining += 1
                self.cooldown_counter = 0

        if not self._is_action_valid(spec):
            self.action_valid = False
        else:
            self.action_valid = True
            cost = self._action_cost(spec)
            if spec.type == ActionType.WATER_DROP:
                self._apply_water_drop(spec)
                self.water_drops_remaining -= 1
                self._dispatch_aircraft()
            elif spec.type == ActionType.FIREBREAK:
                self._apply_firebreak(spec)
            elif spec.type == ActionType.FIRE_TRUCK:
                self._apply_fire_truck(spec)
                self._dispatch_fire_truck()
            self.budget_remaining -= cost

        self._apply_drying()

        fuel_before_spread = float(np.sum(self.fuel, dtype=np.float64))
        burn_rate_before_spread = float(np.sum(self.burn_rate, dtype=np.float64))
        burn_rate_suppressed = max(0.0, burn_rate_before_action - burn_rate_before_spread)
        self._spread_fire()

        active_fires = np.sum(self.grid == CellState.BURNING)
        healthy_trees = np.sum(self.grid == CellState.HEALTHY)

        terminated = active_fires == 0 or healthy_trees == 0
        truncated = self.current_step >= self.max_steps

        self.failure_reason = None
        if terminated and healthy_trees == 0:
            self.failure_reason = "no_healthy"

        self._tick_vehicle_timers()

        reward, components, metrics = self._calculate_reward(
            fuel_before_action=fuel_before_action,
            fuel_before_spread=fuel_before_spread,
            budget_before=budget_before,
            fires_before=fires_before,
            active_fires=active_fires,
            terminated=terminated,
            truncated=truncated,
            burn_rate_suppressed=burn_rate_suppressed,
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

            if info["wind_direction"] in ["NONE", "CALM"] or info["wind_speed"] == 0.0:
                wind_str = "CALM (0.0)"
            else:
                wind_str = f"{info['wind_direction']} ({info['wind_speed']:.1f})"

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
                wind_field=info.get("wind_vector_field"),
                rain_field=info.get("rain_intensity_field"),
            )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()