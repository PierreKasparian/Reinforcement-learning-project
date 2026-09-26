import time
import pygame
from forest_fire_env import ForestFireEnv

if __name__ == "__main__":
    # 1. Instantiate the environment with render_mode="human"
    env = ForestFireEnv(render_mode="human", grid_size=15, p_spread=0.1, max_steps=100)
    obs, info = env.reset()

    print(f"Observation Tensor Shape: {obs.shape}")
    print(f"Initial Info: {info}")

    total_reward = 0
    done = False

    # Main Simulation Loop
    while not done:
        # Sample a random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the current step
        env.render()
        
        # Control simulation speed (e.g., 0.1s delay per step)
        time.sleep(0.5)
        
        total_reward += reward
        done = terminated or truncated

    print(f"Episode Finished! Total Reward: {total_reward:.2f}")
    print(f"Final Info: {info}")
    print("Simulation completed. Close the Pygame window to exit.")

    # Keep window open until manually closed by user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        time.sleep(0.05)

    # Clean up the Pygame window
    env.close()