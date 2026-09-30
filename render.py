import numpy as np
import pygame


class ForestFireRenderer:
    """Handles Pygame visualization for the Forest Fire grid environment."""

    def __init__(self, grid_size=15, window_size=600, render_fps=10):
        self.grid_size = grid_size
        self.window_size = window_size
        self.cell_size = self.window_size // self.grid_size
        self.render_fps = render_fps

        # Color mapping for grid cell states
        self.colors = {
            0: (75, 70, 70),    # Empty / Burned (Burnt Ash)
            1: (34, 139, 34),   # Unburned Tree (Forest Green)
            2: (215, 60, 0),    # Active Fire (Fire Red/Orange)
            3: (105, 105, 105), # Firebreak Barrier (Dark Gray)
            4: (30, 144, 255),  # Water / Retardant (Dodger Blue)
        }

        self.window = None
        self.clock = None

    def render(self, grid, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Forest Fire Overview & Dispatcher")

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Render Grid Cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_val = grid[r, c]
                color = self.colors.get(cell_val, (0, 0, 0))
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)  # Grid border lines

        if mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.render_fps)
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None