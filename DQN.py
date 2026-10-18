"""
Axel Bechtel
Jesper Wentzell
Pieree Kasparian
Ahmet Alper Basturk
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from buffer import ReplayBuffer, device
from model import WildfireCNNQNetwork

class DQNAgent:
    def __init__(self, action_size, grid_size=15, num_spatial=5, num_global=6):
        self.action_size = action_size
        
        self.q_network = WildfireCNNQNetwork(action_size, grid_size, num_spatial, num_global).to(device)
        self.target_network = WildfireCNNQNetwork(action_size, grid_size, num_spatial, num_global).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(100000)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # DQN stability parameters
        self.warmup_steps = 2000
        self.train_freq = 2
        self.target_update_steps = 1000
        self.total_steps = 0
        
    def act(self, state, action_mask=None):
        if random.random() < self.epsilon:
            if action_mask is not None:
                # Group valid actions by type so random exploration selects uniformly among action types rather than flat indices
                g2 = (self.action_size - 1) // 5
                type_slices = [
                    np.array([0]) if action_mask[0] else np.array([], dtype=int),
                    np.where(action_mask[1 : 1 + 2 * g2])[0] + 1,
                    np.where(action_mask[1 + 2 * g2 : 1 + 4 * g2])[0] + 1 + 2 * g2,
                    np.where(action_mask[1 + 4 * g2 :])[0] + 1 + 4 * g2,
                ]
                non_empty_types = [arr for arr in type_slices if len(arr) > 0]
                if len(non_empty_types) > 0:
                    chosen_group = random.choice(non_empty_types)
                    return int(random.choice(chosen_group))
            return random.randint(0, self.action_size - 1)
        
        sp_tensor = torch.tensor(np.array(state["spatial"]), dtype=torch.float32).unsqueeze(0).to(device)
        gl_tensor = torch.tensor(np.array(state["global"]), dtype=torch.float32).unsqueeze(0).to(device)
        
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(sp_tensor, gl_tensor)
            
            if action_mask is not None:
                # Mask out illegal actions by setting their Q-values to an extremely low penalty
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)
                q_values[0, ~mask_tensor] = -1e8
                
        self.q_network.train()
        return int(torch.argmax(q_values).item())
        
    def learn(self):
        self.total_steps += 1
        if len(self.memory) < self.warmup_steps:
            return
        if self.total_steps % self.train_freq != 0:
            return
            
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)
        sp_states, gl_states = states
        next_sp_states, next_gl_states = next_states
        
        q_values = self.q_network(sp_states, gl_states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_sp_states, next_gl_states)
            next_q_values[~next_masks] = -1e8

            # Compute DQN bootstrap target: y_t = r + gamma * (1 - done) * max_a Q_target(s', a)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Fit Q-values to targets using Huber loss (Smooth L1 Loss)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.total_steps % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, filepath):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint.get('total_steps', 0)
