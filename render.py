import numpy as np
import pygame


class ForestFireRenderer:
    def __init__(self, grid_size=15, window_size=600, render_fps=10, max_moisture=100.0):
        self.grid_size = grid_size
        self.window_size = window_size
        self.cell_size = self.window_size // self.grid_size
        self.render_fps = render_fps
        self.max_moisture = max_moisture

        self.static_colors = {
            0: (50, 50, 50),     # Burnt Out
            3: (105, 105, 105),  # Firebreak
            4: (30, 144, 255),   # Fully Saturated WET
        }

        self.LOW_FIRE_COLOR = np.array([255, 220, 0])
        self.HIGH_FIRE_COLOR = np.array([180, 0, 0])
        self.HEALTHY_BASE = np.array([34, 139, 34])
        self.WATER_TINT = np.array([100, 200, 255])

        self.window = None
        self.clock = None
        self.font = None

    def _get_fire_color(self, burn_rate, max_expected_rate=20.0):
        norm_rate = np.clip(burn_rate / max_expected_rate, 0.0, 1.0)
        color = (1.0 - norm_rate) * self.LOW_FIRE_COLOR + norm_rate * self.HIGH_FIRE_COLOR
        return tuple(color.astype(int))

    def _get_healthy_color(self, moisture_level):
        norm_m = np.clip(moisture_level / self.max_moisture, 0.0, 1.0)
        color = (1.0 - norm_m) * self.HEALTHY_BASE + norm_m * self.WATER_TINT
        return tuple(color.astype(int))

    def render(self, grid, burn_rates, moisture, wind_direction="NORTH", wind_speed=1.0, mode="human"):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Forest Fire Overview - Dynamic Wind & Moisture")
            self.font = pygame.font.SysFont("Arial", 16, bold=True)

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                state_val = grid[r, c]

                if state_val == 2:
                    color = self._get_fire_color(burn_rates[r, c])
                elif state_val == 1:
                    color = self._get_healthy_color(moisture[r, c])
                else:
                    color = self.static_colors.get(state_val, (0, 0, 0))

                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        # Draw Wind HUD Overlay
        if mode == "human" and self.font is not None:
            if wind_direction in ["NONE", "CALM"] or wind_speed == 0.0:
                wind_text = "WIND: CALM (0.0)"
            else:
                wind_text = f"WIND: {wind_direction} (Speed: {wind_speed:.1f})"

            text_surface = self.font.render(wind_text, True, (0, 0, 0), (255, 255, 255))
            canvas.blit(text_surface, (10, 10))

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