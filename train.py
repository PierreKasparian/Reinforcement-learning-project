import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from DQN import DQNAgent
from DDQN import DDQNAgent
from env import ForestFireEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Wildfire Environment")
    parser.add_argument('--algo', type=str, default='ddqn', choices=['dqn', 'ddqn'], 
                        help='Algorithm to train: dqn or ddqn')
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--grid-size', type=int, default=15)
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    
    # Environment Setup
    env = ForestFireEnv(grid_size=args.grid_size, max_steps=100)
    action_size = env.action_space.n 
    num_channels = 4
    
    # Initialize Selected Agent
    AgentClass = DDQNAgent if args.algo == 'ddqn' else DQNAgent
    agent = AgentClass(
        action_size=action_size, 
        grid_size=args.grid_size, 
        num_channels=num_channels
    )
    
    # Training Loop
    target_update_freq = 10
    save_freq = 500
    episode_rewards = []
    moving_averages = []
    
    print(f"Starting {args.algo.upper()} training on Wildfire Environment...")
    for episode in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        if episode % target_update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
            
        # Logging
        episode_rewards.append(total_reward)
        moving_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        moving_averages.append(moving_avg)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1:3d} | Reward: {total_reward:7.2f} | 100-ep Avg: {moving_avg:7.2f} | Epsilon: {agent.epsilon:.3f}")
            
        # Checkpointing
        if (episode + 1) % save_freq == 0:
            agent.save_checkpoint(f"./checkpoints/{args.algo}_wildfire_ep{episode + 1}.pth")
            agent.memory.save(f"./checkpoints/{args.algo}_buffer_ep{episode + 1}.pkl")
            
    env.close()
    
    # Visualization
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