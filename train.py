import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQNAgent
from DDQN import DDQNAgent
from rainbow import RainbowAgent
from env import ForestFireEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Wildfire Environment")
    parser.add_argument('--algo', type=str, default='rainbow', choices=['dqn', 'ddqn', 'rainbow'], 
                        help='Algorithm to train: dqn, ddqn, or rainbow')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--grid-size', type=int, default=30)
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    
    # Environment Setup
    env = ForestFireEnv(grid_size=args.grid_size, max_steps=100)
    action_size = env.action_space.n 
    num_channels = 11  
    
    # Initialize Selected Agent
    if args.algo == 'rainbow':
        agent = RainbowAgent(action_size, args.grid_size, num_channels)
    elif args.algo == 'ddqn':
        agent = DDQNAgent(action_size, args.grid_size, num_channels)
    else:
        agent = DQNAgent(action_size, args.grid_size, num_channels)
        
    # Training Loop
    target_update_freq = 10
    save_freq = 500
    episode_rewards = []
    moving_averages = []
    
    print(f"Starting {args.algo.upper()} training on Wildfire Environment ({args.grid_size}x{args.grid_size})...")
    for episode in range(args.episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            mask = env.action_masks()
            action = agent.act(state, action_mask=mask)
            
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_mask = env.action_masks()
            
            agent.memory.push(state, action, reward, next_state, done, next_mask)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
        # Standard Epsilon Decay for DQN/DDQN (Ignored completely by Rainbow)
        if hasattr(agent, 'epsilon'):
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
            
        # Logging
        episode_rewards.append(total_reward)
        moving_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        moving_averages.append(moving_avg)
        
        if (episode + 1) % 10 == 0:
            outcome = next_info.get("reward_metrics", {}).get("outcome", "unknown")
            eps_str = f" | Epsilon: {agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else ""
            print(f"Episode: {episode + 1:4d} | Reward: {total_reward:7.2f} | 100-ep Avg: {moving_avg:7.2f}{eps_str} | Outcome: {outcome}")
            
        # Checkpointing
        if (episode + 1) % save_freq == 0:
            agent.save_checkpoint(f"./checkpoints/{args.algo}_wildfire_ep{episode + 1}.pth")
            
    env.close()
    
    # 4. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Episode Reward', alpha=0.4, color='royalblue')
    plt.plot(moving_averages, label='100-Episode Moving Avg', linewidth=2, color='darkorange')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{args.algo.upper()} Wildfire Training Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = f'./plots/{args.algo}_wildfire_training.png'
    plt.savefig(plot_path)
    print(f"Training plot saved to {plot_path}")

if __name__ == "__main__":
    main()