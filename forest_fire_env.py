import enum
from dataclasses import dataclass
from typing import Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CellState(enum.IntEnum):
    BURNT = 0
    HEALTHY = 1
    BURNING = 2
    FIREBREAK = 3
    WET = 4

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


class ForestFireEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        grid_size=15,
        p_spread=0.1,
        max_steps=100,
        max_fuel=100.0,
        max_moisture=100.0,
        base_burn_rate=2.0,
        burn_acceleration=1.5,
        wind_speed=1.0,
        wind_factor=0.08,  # Scaling factor for wind influence on spread
        rain_intensity: float = 5.0,
        rain_prob: float = 0.2,
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
        self.burn_acceleration = burn_acceleration

        # Wind dynamics
        self.wind_factor = wind_factor
        self.default_wind_speed = wind_speed
        self.wind = Wind(direction=Direction.NORTH, speed=self.default_wind_speed)

        #Rain dynamics
        self.rain_intensity = rain_intensity
        self.rain = Rain(intensity=rain_intensity, rain_prob=rain_prob)

        # State matrices
        self.grid = None
        self.fuel = None
        self.moisture = None
        self.burn_rate = None
        self.current_step = 0

        # Observation Space: 6 channels (Fuel, Moisture, Intensity, Firebreak, Wet, Wind Channel)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(1)

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

        # Initialize wind
        is_active = bool(self.np_random.random() < 0.6)  # 60% chance starting with wind
        initial_dir = Direction(self.np_random.integers(0, 4)) if is_active else Direction.NONE

        self.wind = Wind(
            direction=initial_dir,
            speed=self.default_wind_speed,
            is_active=is_active,
            wind_prob=0.6,
        )

        center = self.grid_size // 2
        self.grid[center, center] = CellState.BURNING
        self.burn_rate[center, center] = self.base_burn_rate

        return self._get_obs(), self._get_info()

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

        if self.grid[r, c] == CellState.BURNING:
            suppression = amount * 0.1
            self.burn_rate[r, c] = max(0.5, self.burn_rate[r, c] - suppression)

        if self.moisture[r, c] >= self.max_moisture:
            self.grid[r, c] = CellState.WET
            self.burn_rate[r, c] = 0.0

    def _update_environmental_factors(self):
        """Step environmental forces like Wind and Rain."""
        # Rain & apply moisture
        self.rain.step(self.np_random)

        if self.rain.is_active:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    added_moisture = self.rain.get_moisture_increase(r, c, (self.grid_size, self.grid_size))
                    if added_moisture > 0:
                        # Uses your existing moisture method/array
                        self.moisture[r, c] = min(
                            self.max_moisture, 
                            self.moisture[r, c] + added_moisture
                        )

        # Wind
        self.wind.step(self.np_random, transition_prob=0.15)

    def _get_obs(self):
        obs = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)

        obs[0] = self.fuel / self.max_fuel
        obs[1] = self.moisture / self.max_moisture
        max_expected_rate = self.base_burn_rate + (self.burn_acceleration * 20.0)
        obs[2] = np.clip(self.burn_rate / max_expected_rate, 0.0, 1.0)
        obs[3] = (self.grid == CellState.FIREBREAK).astype(np.float32)
        obs[4] = (self.grid == CellState.WET).astype(np.float32)
        
        # Channel 5: Wind direction indicator (4.0 / 4.0 = 1.0 for CALM / NONE)
        obs[5] = self.wind.direction.value / 4.0

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
                self.burn_rate[r, c] += self.burn_acceleration

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

    def step(self, action):
        self.current_step += 1
        self._update_environmental_factors()
        self._spread_fire()

        active_fires = np.sum(self.grid == CellState.BURNING)
        healthy_trees = np.sum(self.grid == CellState.HEALTHY)

        terminated = active_fires == 0 or healthy_trees == 0
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), 0.0, terminated, truncated, self._get_info()

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
                mode=self.render_mode,
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()