import time
import pygame
from forest_fire_env import ForestFireEnv

# CONFIGURATION & SIMULATION PARAMETERS

# Environment Setup
GRID_SIZE = 30
MAX_STEPS = 200
SIMULATION_DELAY = 0.10  # Seconds per step (lower = faster)
LOG_FILE_PATH = "simulation_log.txt"

# Fire Dynamics
P_SPREAD = 0.1           # Base fire spread probability
MAX_FUEL = 100.0         # Maximum fuel per cell
MAX_MOISTURE = 100.0     # Moisture cap per cell
BASE_BURN_RATE = 2.0     # Initial burn rate for new fires
BURN_ACCELERATION = 1.5  # Rate of fire intensity increase per step

# Wind Mechanics
WIND_SPEED = 1.0         # Wind velocity scalar
WIND_FACTOR = 1.0        # Spread impact multiplier (1.0 = +100% downwind, -100% upwind)

# Rain Mechanics
START_WITH_RAIN = True   # Force rain ON at step 0 (True / False)
RAIN_INTENSITY = 0.2     # Moisture units added per cell per step during rain
RAIN_PROB = 0.5          # Probability of rain naturally starting/stopping per check


if __name__ == "__main__":
    # Instantiate Environment using central configuration parameters
    env = ForestFireEnv(
        render_mode="human",
        grid_size=GRID_SIZE,
        p_spread=P_SPREAD,
        max_steps=MAX_STEPS,
        max_fuel=MAX_FUEL,
        max_moisture=MAX_MOISTURE,
        base_burn_rate=BASE_BURN_RATE,
        burn_acceleration=BURN_ACCELERATION,
        wind_speed=WIND_SPEED,
        wind_factor=WIND_FACTOR,
        rain_intensity=RAIN_INTENSITY,
        rain_prob=RAIN_PROB,
    )

    obs, info = env.reset()

    # Apply initial rain toggle preference
    env.rain.is_active = START_WITH_RAIN

    with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_file:
        log_file.write("--- Forest Fire Simulation (Configurable Setup) ---\n")
        log_file.write(f"Grid Size: {GRID_SIZE}x{GRID_SIZE} | Base Spread Prob: {P_SPREAD}\n")
        log_file.write(f"Wind Factor: {WIND_FACTOR} | Initial Rain State: {START_WITH_RAIN}\n\n")

        print(f"Simulation running... Telemetry logging to '{LOG_FILE_PATH}'")
        print("Controls: Press [R] in the Pygame window to toggle Rain ON/OFF.")

        step_count = 0
        running = True

        while running:
            step_count += 1

            # Advance environmental state & fire spread
            env._update_environmental_factors()
            obs, reward, terminated, truncated, info = env.step(0)
            env.render()

            # Retrieve step telemetry
            healthy = info["healthy_count"]
            burning = info["burning_count"]
            burnt = info["burnt_count"]
            wet = info["wet_count"]
            wind_dir = info["wind_direction"]
            wind_spd = info["wind_speed"]
            is_raining = info.get("rain_active", False)
            rain_val = info.get("rain_intensity", 0.0)

            rain_status = f"ACTIVE ({rain_val:.1f})" if is_raining else "OFF"

            # Format and record step log
            log_line = (
                f"Step {step_count:3d} | "
                f"Burning: {burning:3d} | "
                f"Healthy: {healthy:4d} | "
                f"Wet/Immune: {wet:3d} | "
                f"Burnt: {burnt:4d} | "
                f"Rain: {rain_status:<12s} | "
                f"Wind: {wind_dir} ({wind_spd:.1f})\n"
            )

            log_file.write(log_line)
            log_file.flush()

            # Handle Pygame interactive events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # Press 'R' to manually toggle rain ON/OFF live during simulation
                    if event.key == pygame.K_r:
                        env.rain.is_active = not env.rain.is_active
                        print(f"[Step {step_count}] Manual Override: Rain toggled to {env.rain.is_active}")

            if terminated or truncated:
                running = False

            time.sleep(SIMULATION_DELAY)

    print(f"Simulation finished. Log saved to '{LOG_FILE_PATH}'.")
    env.close()