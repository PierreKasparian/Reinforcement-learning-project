import time
import argparse
import os
import torch
import pygame
from forest_fire_env import ForestFireEnv
from DQN import DQNAgent
from DDQN import DDQNAgent
from rainbow import RainbowAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained RL Agent on Wildfire Environment")
    parser.add_argument('--algo', type=str, default='rainbow', choices=['dqn', 'ddqn', 'rainbow'],
                        help='Algorithm of saved checkpoint')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/rainbow_wildfire_ep500.pth',
                        help='Path to saved agent checkpoint (.pth file)')
    parser.add_argument('--grid-size', type=int, default=30,
                        help='Grid size matching the training run')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Delay in seconds per step for Pygame visual playback')
    parser.add_argument('--save-frames', action='store_true',
                        help='Whether to save simulation frames as images')
    return parser.parse_args()

def main():
    args = parse_args()

    frame_dir = "./sim_frames"
    if args.save_frames:
        os.makedirs(frame_dir, exist_ok=True)
        for f in os.listdir(frame_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(frame_dir, f))

    # Environment Setup
    env = ForestFireEnv(render_mode="human", grid_size=args.grid_size, max_steps=10000)
    action_size = env.action_space.n
    num_channels = env.observation_space.shape[0]

    # Instantiate Agent
    if args.algo == 'rainbow':
        agent = RainbowAgent(action_size, args.grid_size, num_channels)
    elif args.algo == 'ddqn':
        agent = DDQNAgent(action_size, args.grid_size, num_channels)
    else:
        agent = DQNAgent(action_size, args.grid_size, num_channels)

    # Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    agent.load_checkpoint(args.checkpoint)

    # Force pure exploitation mode (zero random exploration)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0

    # Ensure network evaluation mode
    agent.q_network.eval()

    # Run Evaluation Episode
    state, info = env.reset()
    total_reward = 0.0
    step_count = 0
    done = False

    print("Starting visual simulation... Close Pygame window or press CTRL+C to stop.")

    while not done:
        step_count += 1

        # Fetch legal action mask and select greedy action
        mask = env.action_masks()
        action = agent.act(state, action_mask=mask)

        # Step environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render visual frame using ForestFireRenderer
        env.render()

        if args.save_frames:
                    frame_path = os.path.join(frame_dir, f"frame_{step_count:04d}.png")
                    current_screen = pygame.display.get_surface()
                    if current_screen is not None:
                        pygame.image.save(current_screen, frame_path)

        state = next_state
        total_reward += reward

        # Handle Pygame exit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        time.sleep(args.delay)

    outcome = info.get("reward_metrics", {}).get("outcome", "unknown")
    print(f"\n--- Evaluation Complete ---")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Outcome: {outcome}")

    if args.save_frames:
        print(f"Frames saved to {frame_dir}/. You can compile them into a video using ffmpeg:")
        print(f"ffmpeg -framerate 10 -i {frame_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p wildfire_sim.mp4")

    env.close()

if __name__ == "__main__":
    main()