import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class ForestFireEnv(gym.Env):

    """
    Grid Cells:
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
        self.p_spread = p_spread  # Probability of fire spreading to adjacent non-diagonal cells
        self.max_steps = max_steps

        # Grid state mappings: 0=Empty/Burnt, 1=Tree, 2=Fire, 3=Firebreak, 4=Water
        # Define color palette (RGB)

        self.colors = {
            0: (75, 70, 70),    # Empty / Burned (Burnt Ash)
            1: (34, 139, 34),   # Unburned Tree (Forest Green)
            2: (215, 60, 0),    # Active Fire (Fire Orange/Red)
            3: (105, 105, 105), # Firebreak Barrier (Dark Gray)
            4: (30, 144, 255),  # Water Protected (Dodger Blue)
        }

        # Pygame rendering variables
        self.window_size = 600  # 600x600 pixels
        self.cell_size = self.window_size // self.grid_size
        self.window = None
        self.clock = None

        # Define Gym Spaces
        self.action_space = spaces.Discrete(7)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Forest Fire Management RL Environment")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw Grid Cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_val = self.grid[r, c]
                color = self.colors.get(cell_val, (0, 0, 0))
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)  # Grid border lines

        # Draw Agent (Firefighter)
        center_x = int((self.agent_pos[1] + 0.5) * self.cell_size)
        center_y = int((self.agent_pos[0] + 0.5) * self.cell_size)
        radius = self.cell_size // 3

        # Colors
        RED = (200, 30, 30)             # Visor
        YELLOW = (255, 215, 0)          # Helmet / Reflective Stripes
        SAFETY_ORANGE = (255, 90, 0)    # Coat
        BLACK = (0, 0, 0)               # Outline

        # Base Body (Red Suit)
        pygame.draw.circle(canvas, SAFETY_ORANGE, (center_x, center_y), radius)

        # Reflective Safety Stripe
        stripe_rect = pygame.Rect(
            center_x - radius, 
            center_y - (radius // 5), 
            radius * 2, 
            radius // 2
        )
        pygame.draw.rect(canvas, YELLOW, stripe_rect)

        # Firefighter Helmet Top
        helmet_rect = pygame.Rect(
            center_x - (radius // 1.3),
            center_y - radius,
            radius * 1.5,
            radius
        )
        pygame.draw.ellipse(canvas, YELLOW, helmet_rect)

        # Visor
        pygame.draw.line(
            canvas, 
            RED, 
            (center_x - radius, center_y - (radius // 4)), 
            (center_x + radius, center_y - (radius // 4)), 
            4
        )

        # Outer Black Outline
        pygame.draw.circle(canvas, BLACK, (center_x, center_y), radius, 2)

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.init()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize full forest
        self.grid = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        self.fire_age = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Starts fire at the center of the grid (2: Fire)
        center = self.grid_size // 2
        self.grid[center, center] = 2

        # Spawns Agent at top-left corner (0, 0)
        self.agent_pos = np.array([0, 0], dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        # Constructs the 4-channel spatial state tensor for CNN input.
        obs = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Fire locations
        obs[0] = (self.grid == 2).astype(np.float32)

        # Channel 1: Fuel / Unburned Trees
        obs[1] = (self.grid == 1).astype(np.float32)

        # Channel 2: Firebreak locations
        obs[2] = (self.grid == 3).astype(np.float32)

        # Channel 3: Agent position
        obs[3, self.agent_pos[0], self.agent_pos[1]] = 1.0

        return obs

    def _get_info(self):
        return {
            "unburned_count": int(np.sum(self.grid == 1)),
            "fire_count": int(np.sum(self.grid == 2)),
            "firebreak_count": int(np.sum(self.grid == 3)),
        }

    def _spread_fire(self):

        # Find all currently burning cells
        burning_coords = np.argwhere(self.grid == 2)

        # Store new fires and new burnt cells to apply simultaneously at the end of the step
        new_fires = set()
        new_burnt = set()

        for r, c in burning_coords:

            # Increment age for this burning cell
            self.fire_age[r, c] += 1

            # Define 4-directional neighbors
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

            # Check if cell reached maximum burning duration (5 steps)
            if self.fire_age[r, c] >= 10:
                # 1. Force-ignite all adjacent healthy trees
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if self.grid[nr, nc] == 1:  # Healthy tree
                            new_fires.add((nr, nc))

                # Burn out this cell completely
                new_burnt.add((r, c))

            else:

                # Normal probabilistic spread for fires under n steps
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if (
                            self.grid[nr, nc] == 1
                            and self.np_random.random() < self.p_spread
                        ):
                            new_fires.add((nr, nc))

        # Apply changes to grid
        for r, c in new_burnt:
            self.grid[r, c] = 0  # Set to Empty/Burnt Ash (0)
            self.fire_age[r, c] = 0

        for r, c in new_fires:
            self.grid[r, c] = 2  # Set to burning
            if self.fire_age[r, c] == 0:
                self.fire_age[r, c] = 1

    def step(self, action):
        self.current_step += 1

        # Execute Agent Action
        reward = -0.1  # Small step penalty to encourage speed

        if action == 0:  # Move North
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Move South
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Move East
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 3:  # Move West
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 4:  # Build Firebreak at current position
            r, c = self.agent_pos

            # If current tile has trees or fire, convert it to a firebreak
            if self.grid[r, c] in [1, 2]:
                self.grid[r, c] = 3
                reward += 1.0  # Reward successful placement of a barrier

        # Simulate Fire Spread (Cellular Automata)
        self._spread_fire()

        # Calculate Environmental Rewards
        unburned_trees = np.sum(self.grid == 1)
        active_fires = np.sum(self.grid == 2)

        # Reward proportional to remaining forest saved
        reward += float(unburned_trees) * 0.01

        # 4. Termination Conditions
        terminated = False
        if active_fires == 0:
            terminated = True
            reward += 50.0  # Big reward for extinguishing/containing the fire
        elif unburned_trees == 0:
            terminated = True
            reward -= 50.0  # Penalty if whole forest burns down

        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()