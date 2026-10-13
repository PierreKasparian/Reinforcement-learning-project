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
    @property
    def speed(self) -> float:
        return self.current_speed if self.is_active else 0.0

    @property
    def direction(self):
        if not self.is_active or self.current_speed == 0.0:
            return Direction.NONE
        # Map the continuous angle (radians) to a cardinal Direction enum for UI compatibility
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
                # Pick a random center point for the localized wind pocket/front
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

        # Drift the patch center slightly over time for organic movement
        if hasattr(self, 'patch_center'):
            self.patch_center += np_random.uniform(-0.5, 0.5, size=2)
            # Keep within grid bounds
            self.patch_center = np.clip(self.patch_center, 0, self.grid_size)
        else:
            self.patch_center = np.array([self.grid_size / 2.0, self.grid_size / 2.0], dtype=np.float32)

        # Gradually shift direction and speed
        self.current_direction_angle += np_random.uniform(-0.25, 0.25)
        self.current_speed = np.clip(self.current_speed + np_random.uniform(-0.2, 0.2), 0.5, self.max_speed)

        base_dx = np.cos(self.current_direction_angle) * self.current_speed
        base_dy = np.sin(self.current_direction_angle) * self.current_speed

        target_field = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        
        # Define a localized radius for the wind pocket
        pocket_radius = self.grid_size * 0.4

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Calculate distance from this cell to the moving wind pocket center
                dist = np.hypot(r - self.patch_center[0], c - self.patch_center[1])
                
                # Smooth falloff so wind is strongest near the center and fades out to 0 at the edges
                if dist < pocket_radius:
                    spatial_mod = 0.5 * (1.0 + np.cos(np.pi * dist / pocket_radius))
                else:
                    spatial_mod = 0.0  # Calm outside the localized wind pocket!

                target_field[r, c, 0] = base_dx * spatial_mod
                target_field[r, c, 1] = base_dy * spatial_mod

        # Smooth temporal blending
        self.vector_field = 0.8 * self.vector_field + 0.2 * target_field

    def get_vector_at(self, r: int, c: int) -> Tuple[float, float]:
        """Returns the localized (dx, dy) wind vector at cell (r, c)."""
        if not self.is_active and np.all(self.vector_field == 0):
            return 0.0, 0.0
        return float(self.vector_field[r, c, 0]), float(self.vector_field[r, c, 1])

class Rain:
    def __init__(
        self,
        intensity: float = 5.0,     # Moisture units added per step during rain
        is_active: bool = True,
        rain_prob: float = 0.2,      # Chance of rain starting/stopping
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

    fire_damage: float = 1.5
    action_cost: float = 0.01
    time: float = 0.005
    success: float = 1.0
    failure: float = 0.5


class ForestFireEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        grid_size=15,
        p_spread=0.045,
        max_steps=300,
        max_fuel=100.0,
        max_moisture=100.0,
        base_burn_rate=2.0,
        burn_growth_rate = 0.35,
        max_burn_rate = 30.0,
        wind_speed=1.0,
        wind_factor=0.05,
        rain_intensity=1.0,
        rain_prob=0.1,
        budget_replenish_rate=0.5,
        water_drop_cooldown=5,
        cooldown_counter=0,
        aircraft_turnaround_steps=4,
        num_fire_trucks=2,
        truck_turnaround_steps=3,
        # Set base resources for a 15x15 grid for dynamic scaling
        max_water_drops=12,
        max_firebreak_capacity=15,
        max_budget=150,
        action_costs=None,
        water_drop_amount=50.0,
        truck_amount=35.0,
        dry_rate=0.5,
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
        self.max_firebreak_capacity = int(max_firebreak_capacity * area_ratio)
        self.max_budget = int(max_budget * area_ratio)
        
        # Dynamically scale the size of the tools so they remain useful on larger grids
        self.line_half_length = max(2, self.grid_size // 6)

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
        self.budget_replenish_rate = budget_replenish_rate
        self.water_drop_cooldown = water_drop_cooldown
        self.cooldown_counter = cooldown_counter

        # Vehicle turnaround: after a dispatch, a vehicle is busy (flying back to
        # refill / driving back) and becomes available again N steps later.
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
        self.initial_total_fuel = float(np.sum(self.fuel, dtype=np.float64))
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

        # Initialize rain
        self.rain = Rain(
            intensity=self.rain_intensity,
            rain_prob=self.rain_prob,
            is_active=False,
        )

        # Suppression resources
        self.water_drops_remaining = self.max_water_drops
        self.firebreak_capacity = self.max_firebreak_capacity
        self.cooldown_counter = 0
        self.aircraft_cooldown = 0
        self.truck_timers = np.zeros(self.num_fire_trucks, dtype=np.int32)
        self._sync_vehicle_availability()
        self.budget_remaining = self.max_budget

        self.last_action = ActionSpec(ActionType.NOOP)
        self.action_valid = True
        self.failure_reason = None

        center = self.grid_size // 2
        spawn_radius = max(1, self.grid_size // 6)
        
        # Randomly pick coordinates within the central bounding box
        r = self.np_random.integers(center - spawn_radius, center + spawn_radius + 1)
        c = self.np_random.integers(center - spawn_radius, center + spawn_radius + 1)
        
        self.grid[r, c] = CellState.BURNING
        self.burn_rate[r, c] = self.base_burn_rate

        min_burn_steps = 1
        max_burn_steps = max(3, self.grid_size // 4)
        initial_burn_steps = self.np_random.integers(min_burn_steps, max_burn_steps + 1)

        for _ in range(initial_burn_steps):
            self._update_environmental_factors()
            self._apply_drying()
            self._spread_fire()

        return self._get_obs(), self._get_info()

    # ------- Environment -------

    def get_wind_vector_at(self, r: int, c: int) -> Tuple[float, float]:
        """Returns normalized (dr, dc) wind vector at cell (r, c)."""
        if not self.wind.is_active or self.wind.direction == Direction.NONE:
            return 0.0, 0.0
        return DIRECTION_OFFSETS[self.wind.direction]

    def _calculate_wind_modifier(self, source_r: int, source_c: int, target_r: int, target_c: int) -> float:
        """Calculates spread multiplier using localized, smoothly varying wind vectors."""
        spread_dr = target_r - source_r
        spread_dc = target_c - source_c

        # Get localized wind vector at the source burning cell
        wind_dx, wind_dy = self.wind.get_vector_at(source_r, source_c)
        
        if wind_dx == 0.0 and wind_dy == 0.0:
            return 1.0

        # Alignment between spread direction and localized wind vector
        alignment = (spread_dr * wind_dx) + (spread_dc * wind_dy)

        modifier = 1.0 + (alignment * self.wind_factor)
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

    # ------- Vehicles -------

    def _sync_vehicle_availability(self):
        """Derive the availability flags from the turnaround timers."""
        self.aircraft_available = self.aircraft_cooldown == 0
        self.fire_truck_available = bool(np.any(self.truck_timers == 0))

    def _dispatch_aircraft(self):
        """The aircraft leaves to refill and returns after its turnaround time."""
        self.aircraft_cooldown = self.aircraft_turnaround_steps
        self._sync_vehicle_availability()

    def _dispatch_fire_truck(self):
        """Send the first idle truck; it returns after its turnaround time."""
        idle = np.flatnonzero(self.truck_timers == 0)
        self.truck_timers[idle[0]] = self.truck_turnaround_steps
        self._sync_vehicle_availability()

    def _tick_vehicle_timers(self):
        """Advance turnaround timers by one step (called at the end of step())."""
        self.aircraft_cooldown = max(0, self.aircraft_cooldown - 1)
        self.truck_timers = np.maximum(0, self.truck_timers - 1)
        self._sync_vehicle_availability()

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
        spatial = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        spatial[0] = self.fuel / self.max_fuel
        spatial[1] = self.moisture / self.max_moisture
        max_expected_rate = self.max_burn_rate
        spatial[2] = np.clip(self.burn_rate / max_expected_rate, 0.0, 1.0)
        spatial[3] = (self.grid == CellState.FIREBREAK).astype(np.float32)
        spatial[4] = (self.grid == CellState.WET).astype(np.float32)
        
        global_obs = np.array([
            self.wind.direction.value / 4.0,
            self.water_drops_remaining / self.max_water_drops,
            self.firebreak_capacity / self.max_firebreak_capacity,
            1.0 - (self.aircraft_cooldown / self.aircraft_turnaround_steps),
            float(np.mean(self.truck_timers == 0)),
            self.budget_remaining / self.max_budget
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
            "resources": {
                "water_drops_remaining": self.water_drops_remaining,
                "firebreak_capacity": self.firebreak_capacity,
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
        # Handle Extinguishing and Fuel Consumption Vectorized
        burning_mask = (self.grid == CellState.BURNING)
        
        # Moisture extinction probabilities
        p_extinguish = np.zeros_like(self.moisture)
        moisture_active = burning_mask & (self.moisture > 0)
        p_extinguish[moisture_active] = self.moisture[moisture_active] / (
            self.moisture[moisture_active] + self.burn_rate[moisture_active] * 10.0
        )
        
        # Random rolls for extinguishing
        rolls = self.np_random.random(size=self.grid.shape)
        extinguished_mask = moisture_active & (rolls < p_extinguish)
        
        self.grid[extinguished_mask] = CellState.HEALTHY
        self.burn_rate[extinguished_mask] = 0.0
        
        # Decrease moisture where not extinguished
        still_burning_mask = burning_mask & ~extinguished_mask
        self.moisture[still_burning_mask] = np.maximum(0.0, self.moisture[still_burning_mask] - 2.0)
        
        # Consume fuel
        self.fuel[still_burning_mask] -= self.burn_rate[still_burning_mask]

        curr_rate = self.burn_rate[still_burning_mask]
        logistic_delta = self.burn_growth_rate * curr_rate * (1.0 - (curr_rate / self.max_burn_rate))
        self.burn_rate[still_burning_mask] = np.minimum(self.max_burn_rate, curr_rate + logistic_delta)
        
        # Burnout (Fuel <= 0)
        burnout_mask = still_burning_mask & (self.fuel <= 0.0)
        self.fuel[burnout_mask] = 0.0
        self.grid[burnout_mask] = CellState.BURNT
        self.burn_rate[burnout_mask] = 0.0

        # Spread to Neighbors 
        new_fires = set()
        
        # Probabilistic spread from cells that just burnt out
        burnout_coords = np.argwhere(burnout_mask)
        for r, c in burnout_coords:
            # Intensity ratio at burnout point (normalized 0 to 1)
            intensity_ratio = self.burn_rate[r, c] / self.max_burn_rate
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr, nc] == CellState.HEALTHY:
                        moisture_factor = 1.0 - (self.moisture[nr, nc] / self.max_moisture)
                        wind_mod = self._calculate_wind_modifier(r, c, nr, nc)
                        
                        # Max intensity has probability 1 * moisture_factor * wind_mod (adjusted by intensity ratio)
                        effective_p = intensity_ratio * moisture_factor * wind_mod
                        
                        if self.np_random.random() < min(1.0, effective_p):
                            new_fires.add((nr, nc))

        # Probabilistic spread from currently active fires
        active_coords = np.argwhere(still_burning_mask & ~burnout_mask)
        for r, c in active_coords:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
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
            "fire_damage": -w.fire_damage * fuel_burned / fuel_scale,
            "action_cost": -w.action_cost * budget_spent / budget_scale,
            "time": -w.time / time_scale if active_fires > 0 else 0.0,
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
            if spec.type == ActionType.WATER_DROP:
                self._apply_water_drop(spec)
                self.water_drops_remaining -= 1
                self._dispatch_aircraft()
            elif spec.type == ActionType.FIREBREAK:
                self._apply_firebreak(spec)
                self.firebreak_capacity -= 1
            elif spec.type == ActionType.FIRE_TRUCK:
                self._apply_fire_truck(spec)
                self._dispatch_fire_truck()
            self.budget_remaining -= self._action_cost(spec)

        self._apply_drying()

        # Separate intervention-related fuel removal from fire damage.
        fuel_before_spread = float(np.sum(self.fuel, dtype=np.float64))
        self._spread_fire()

        active_fires = np.sum(self.grid == CellState.BURNING)
        healthy_trees = np.sum(self.grid == CellState.HEALTHY)
        budget_exhausted = self.budget_remaining == 0 and active_fires > 0

        # TODO: add failure termination when the burned area reaches the grid edge.
        terminated = active_fires == 0 or healthy_trees == 0
        truncated = self.current_step >= self.max_steps

        self.failure_reason = None
        if terminated:
            if healthy_trees == 0:
                self.failure_reason = "no_healthy"

        # Advance vehicle timers last so the returned observation and the next
        # action_masks() describe exactly the state of the next decision.
        self._tick_vehicle_timers()

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
                wind_field=info.get("wind_vector_field"),
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()