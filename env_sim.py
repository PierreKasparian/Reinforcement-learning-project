import time
import pygame
from forest_fire_env import ForestFireEnv

if __name__ == "__main__":
    # Simulation Parameters
    GRID_SIZE = 30
    P_SPREAD = 0.1
    MAX_STEPS = 200
    SIMULATION_DELAY = 0.15  # Seconds between frames

    # Environment Configuration
    MAX_FUEL = 100.0
    MAX_MOISTURE = 100.0
    BASE_BURN_RATE = 2.0
    BURN_ACCELERATION = 1.5
    WIND_SPEED = 1.5
    WIND_FACTOR = 5.0

    # Instantiate Environment
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
    )

    obs, info = env.reset()

    print("--- Starting Forest Fire Simulation (Wind & Moisture Enabled) ---")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Observation Shape: {obs.shape}")  # Should be (6, GRID_SIZE, GRID_SIZE)
    print(f"Initial State: {info}\n")

    step_count = 0
    running = True

    while running:
        step_count += 1

        # Advance natural spread & wind mechanics
        env._spread_fire()

        # Render grid state & wind compass HUD
        env.render()

        # Retrieve updated telemetry
        info = env._get_info()

        healthy = info["healthy_count"]
        burning = info["burning_count"]
        burnt = info["burnt_count"]
        wet = info["wet_count"]
        wind_dir = info["wind_direction"]
        wind_spd = info["wind_speed"]

        print(
            f"Step {step_count:3d} | "
            f"Burning: {burning:3d} | "
            f"Healthy: {healthy:4d} | "
            f"Wet/Immune: {wet:3d} | "
            f"Burnt: {burnt:4d} | "
            f"Wind: {wind_dir} ({wind_spd:.1f})"
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Termination conditions
        if burning == 0:
            print("\nSimulation Ended: All active fires extinguished or burned out.")
            running = False
        elif healthy == 0 and burning == 0:
            print("\nSimulation Ended: Entire forest consumed by fire.")
            running = False
        elif step_count >= MAX_STEPS:
            print(f"\nSimulation Ended: Reached max step limit ({MAX_STEPS}).")
            running = False

        time.sleep(SIMULATION_DELAY)

    print(f"\nFinal Statistics: {info}")
    print("Close the Pygame window to exit.")

    window_open = True
    while window_open:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                window_open = False
        time.sleep(0.05)

    env.close()