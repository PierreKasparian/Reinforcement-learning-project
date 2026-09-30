import time
import pygame
from forest_fire_env import ForestFireEnv

if __name__ == "__main__":
    # Initialize environment parameters
    GRID_SIZE = 30
    P_SPREAD = 0.1
    MAX_STEPS = 200
    SIMULATION_DELAY = 0.2  # Seconds between frames

    # Instantiate the environment
    env = ForestFireEnv(
        render_mode="human",
        grid_size=GRID_SIZE,
        p_spread=P_SPREAD,
        max_steps=MAX_STEPS,
    )

    # Reset environment state
    obs, info = env.reset()

    print("--- Starting Natural Forest Fire Simulation ---")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Spread Probability (p_spread): {P_SPREAD}")
    print(f"Initial State: {info}\n")

    step_count = 0
    running = True

    # Main Unsupervised Simulation Loop
    while running:
        step_count += 1

        # Advance natural cellular automata (fire spread) without agent intervention
        env._spread_fire()

        # Render current grid state
        env.render()

        # Print step progress and stats
        info = env._get_info()
        print(
            f"Step {step_count:3d} | Active Fires: {info['fire_count']:3d} | "
            f"Unburned Trees: {info['unburned_count']:4d} | Burnt Out: {info['grid_size_total'] if 'grid_size_total' in info else (GRID_SIZE**2 - info['fire_count'] - info['unburned_count']):4d}"
        )

        # Check Pygame window close events during execution
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Termination condition: No active fires remaining or max steps reached
        if info["fire_count"] == 0:
            print("\nSimulation Ended: Fire extinguished / burned out naturally.")
            running = False
        elif info["unburned_count"] == 0:
            print("\nSimulation Ended: Entire forest consumed by fire.")
            running = False
        elif step_count >= MAX_STEPS:
            print(f"\nSimulation Ended: Reached max step limit ({MAX_STEPS}).")
            running = False

        # Control simulation speed
        time.sleep(SIMULATION_DELAY)

    print(f"\nFinal Statistics: {info}")
    print("Close the Pygame window to exit.")

    # Keep window open until user closes it
    window_open = True
    while window_open:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                window_open = False
        time.sleep(0.05)

    # Clean up Pygame resources
    env.close()