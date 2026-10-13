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
        
        # Rain visual settings
        self.RAIN_OVERLAY_COLOR = (50, 100, 200, 35)  # Semi-transparent blue tint overlay
        self.RAIN_DROP_COLOR = (180, 220, 255, 150)

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

    def render(
        self,
        grid,
        burn_rates,
        moisture,
        wind_direction="NORTH",
        wind_speed=1.0,
        wind_info=None,
        rain_active=False,
        rain_intensity=0.0,
        resources=None,
        mode="human",
        wind_field=None,
        rain_field=None,
    ):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Forest Fire Overview - Dynamic Wind, Moisture & Rain")
            self.font = pygame.font.SysFont("Arial", 16, bold=True)

        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw Grid Cells
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

        # Draw Wind Vector Overlays (if wind is active and cell size permits)
        if wind_speed > 0.0 and self.cell_size >= 15:
            wind_overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            
            # Sample every few cells or draw on every cell depending on grid size
            step_stride = max(1, self.grid_size // 10)  # e.g., sample roughly 10x10 arrows max for clarity
            for r in range(0, self.grid_size, step_stride):
                for c in range(0, self.grid_size, step_stride):
                    # Center of the cell
                    cx = c * self.cell_size + self.cell_size // 2
                    cy = r * self.cell_size + self.cell_size // 2
                    
                    # If wind_field matrix is passed, use local vector; otherwise use global approximation
                    if wind_field is not None:
                        dx, dy = wind_field[r, c]
                    else:
                        # Fallback approximation based on global wind direction/speed
                        dx, dy = 0.0, 0.0
                        if wind_direction == "NORTH": dy = -wind_speed
                        elif wind_direction == "SOUTH": dy = wind_speed
                        elif wind_direction == "EAST": dx = wind_speed
                        elif wind_direction == "WEST": dx = -wind_speed

                    mag = np.hypot(dx, dy)
                    if mag > 0.05:
                        # Scale arrow length based on magnitude and cell size
                        arrow_len = min(self.cell_size * 0.8, max(4, mag * 4))
                        end_x = cx + int((dx / mag) * arrow_len)
                        end_y = cy + int((dy / mag) * arrow_len)
                        
                        # Draw semi-transparent white/cyan wind vector line
                        pygame.draw.line(wind_overlay, (255, 255, 255, 120), (cx, cy), (end_x, end_y), 1)
                        # Tiny dot at the tip
                        pygame.draw.circle(wind_overlay, (200, 230, 255, 150), (end_x, end_y), 2)

            canvas.blit(wind_overlay, (0, 0))

        # Draw Rain Overlay Effects
        if rain_active and rain_field is not None:
            rain_overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
            ticks = pygame.time.get_ticks()

            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    cell_rain_val = rain_field[r, c]
                    if cell_rain_val > 0.0:
                        # Draw a semi-transparent blue tint over wet/storm cells scaled by intensity
                        alpha = int(np.clip(cell_rain_val * 25, 10, 120))
                        cell_rect = pygame.Rect(
                            c * self.cell_size,
                            r * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        )
                        pygame.draw.rect(rain_overlay, (50, 100, 200, alpha), cell_rect)

                        # Draw subtle rain streaks inside active storm cells if cell size is large enough
                        if self.cell_size >= 12 and (r + c + ticks // 100) % 3 == 0:
                            rx = c * self.cell_size + self.cell_size // 2
                            ry = r * self.cell_size + self.cell_size // 2
                            pygame.draw.line(
                                rain_overlay,
                                self.RAIN_DROP_COLOR,
                                (rx, ry - 4),
                                (rx - 2, ry + 4),
                                1,
                            )

            canvas.blit(rain_overlay, (0, 0))

        # Draw Telemetry HUD Overlay
        if mode == "human" and self.font is not None:
            # Check wind_info first; fallback to direction/speed if wind_info is None
            if wind_info is not None:
                wind_text = f"WIND: {wind_info}"
            elif wind_direction in ["NONE", "CALM"] or wind_speed == 0.0:
                wind_text = "WIND: CALM (0.0)"
            else:
                wind_text = f"WIND: {wind_direction} ({wind_speed:.1f})"

            rain_text = f"RAIN: ACTIVE ({rain_intensity:.1f})" if rain_active else "RAIN: OFF"
            hud_text = f"{wind_text}  |  {rain_text}"

            hud_bg_color = (220, 240, 255) if rain_active else (255, 255, 255)
            text_surface = self.font.render(hud_text, True, (0, 0, 0), hud_bg_color)
            
            # Padding around HUD text box
            padding_rect = text_surface.get_rect(topleft=(10, 10)).inflate(8, 6)
            pygame.draw.rect(canvas, hud_bg_color, padding_rect)
            pygame.draw.rect(canvas, (100, 100, 100), padding_rect, 1)
            canvas.blit(text_surface, (10, 10))

            if resources is not None:
                resources_text = (
                    f"BUDGET: {resources['budget_remaining']}  |  "
                    f"DROPS: {resources['water_drops_remaining']}  |  "
                    f"FIREBREAK: {resources['firebreak_capacity']}  |  "
                    f"AIRCRAFT: {'YES' if resources['aircraft_available'] else str(resources.get('aircraft_cooldown', 0)) + ' steps'}  |  "
                    f"TRUCKS: {resources.get('trucks_idle', int(resources['fire_truck_available']))}/{resources.get('num_fire_trucks', 1)}"
                )
                resources_y = padding_rect.bottom + 6
                resources_surface = self.font.render(resources_text, True, (0, 0, 0), hud_bg_color)
                resources_rect = resources_surface.get_rect(topleft=(10, resources_y)).inflate(8, 6)
                pygame.draw.rect(canvas, hud_bg_color, resources_rect)
                pygame.draw.rect(canvas, (100, 100, 100), resources_rect, 1)
                canvas.blit(resources_surface, (10, resources_y))

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