import gymnasium as gym
from gymnasium import spaces
import numpy as np
from render import ForestFireRenderer


class ForestFireEnv(gym.Env):
    """
    Grid State Values:
    0: Empty / Burned Out
    1: Unburned Trees (Fuel)
    2: Fire
    3: Firebreak
    4: Water
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, grid_size=15, p_spread=0.1, max_steps=100):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.p_spread = p_spread  # Probability of fire spreading to adjacent cells
        self.max_steps = max_steps

        # Grid state representation
        self.grid = None
        self.fire_age = None
        self.current_step = 0

        # Define Observation Space (3 channels: Fire, Unburned Trees, Firebreak/Water)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.grid_size, self.grid_size),
            dtype=np.float32,
        )

        # Placeholder Action Space (Define your dispatch/resource action space here)
        # e.g., MultiDiscrete([grid_size, grid_size, num_resource_types])
        self.action_space = spaces.Discrete(1)

        # Initialize renderer if render_mode is specified
        self.renderer = None
        if self.render_mode is not None:
            self.renderer = ForestFireRenderer(grid_size=self.grid_size, render_fps=self.metadata["render_fps"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize full forest (1: Unburned Trees)
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.fire_age = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Start initial fire at the center of the grid (2: Fire)
        center = self.grid_size // 2
        self.grid[center, center] = 2

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        """Constructs the spatial state tensor for CNN input."""
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Fire locations
        obs[0] = (self.grid == 2).astype(np.float32)

        # Channel 1: Fuel / Unburned Trees
        obs[1] = (self.grid == 1).astype(np.float32)

        # Channel 2: Barriers (Firebreaks & Water)
        obs[2] = np.logical_or(self.grid == 3, self.grid == 4).astype(np.float32)

        return obs

    def _get_info(self):
        return {
            "unburned_count": int(np.sum(self.grid == 1)),
            "fire_count": int(np.sum(self.grid == 2)),
            "firebreak_count": int(np.sum(self.grid == 3)),
            "water_count": int(np.sum(self.grid == 4)),
        }

    def _spread_fire(self):
        """Cellular Automata simulation step for fire propagation."""
        burning_coords = np.argwhere(self.grid == 2)

        new_fires = set()
        new_burnt = set()

        for r, c in burning_coords:
            self.fire_age[r, c] += 1
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

            # Check if cell reached maximum burning duration
            if self.fire_age[r, c] >= 10:
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid[nr, nc] == 1:  # Healthy tree
                            new_fires.add((nr, nc))
                new_burnt.add((r, c))
            else:
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if (
                            self.grid[nr, nc] == 1
                            and self.np_random.random() < self.p_spread
                        ):
                            new_fires.add((nr, nc))

        # Apply state changes to grid
        for r, c in new_burnt:
            self.grid[r, c] = 0  # Empty/Burnt
            self.fire_age[r, c] = 0

        for r, c in new_fires:
            self.grid[r, c] = 2  # Active Fire
            if self.fire_age[r, c] == 0:
                self.fire_age[r, c] = 1

    def step(self, action):
        self.current_step += 1

        # TODO: Apply resource dispatch action to self.grid here
        
        # Advance environment state (Fire spread)
        self._spread_fire()

        active_fires = np.sum(self.grid == 2)
        unburned_trees = np.sum(self.grid == 1)

        # Check termination
        terminated = active_fires == 0 or unburned_trees == 0
        truncated = self.current_step >= self.max_steps

        # Placeholders for reward (to be customized for resource placement strategy)
        reward = 0.0

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.renderer is not None:
            return self.renderer.render(self.grid, mode=self.render_mode)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()