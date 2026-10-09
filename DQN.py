import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from buffer import ReplayBuffer, device
from model import WildfireCNNQNetwork

class DQNAgent:
    def __init__(self, action_size, grid_size=15, num_channels=11):
        self.action_size = action_size
        
        self.q_network = WildfireCNNQNetwork(action_size, grid_size, num_channels).to(device)
        self.target_network = WildfireCNNQNetwork(action_size, grid_size, num_channels).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.memory = ReplayBuffer(100000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
    def act(self, state, action_mask=None):
        if random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    return int(random.choice(valid_actions))
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions so they are never selected
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(device)
                q_values[0, ~mask_tensor] = -float('inf')
                
        return torch.argmax(q_values).item()
        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)
        
        q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            
            # Prevent the target network from evaluating invalid future actions
            next_q_values = next_q_values.masked_fill(~next_masks, -float('inf'))
            
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, filepath):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']