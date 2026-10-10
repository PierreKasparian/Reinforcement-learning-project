import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from buffer import PrioritizedReplayBuffer, device
from model import RainbowCNNQNetwork

class RainbowAgent:
    def __init__(self, action_size, grid_size=30, num_channels=11):
        self.action_size = action_size
        self.batch_size = 64
        self.gamma = 0.99
        
        # Rainbow specific Hyperparameters
        self.n_step = 3
        self.num_atoms = 51
        self.v_min = -300.0  # Tuned for Wildfire reward bounds
        self.v_max = 300.0
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # PER Hyperparameters
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        
        self.q_network = RainbowCNNQNetwork(action_size, grid_size, num_channels, self.num_atoms, self.v_min, self.v_max).to(device)
        self.target_network = RainbowCNNQNetwork(action_size, grid_size, num_channels, self.num_atoms, self.v_min, self.v_max).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.memory = PrioritizedReplayBuffer(100000, alpha=0.6, n_step=self.n_step, gamma=self.gamma)
        
    def act(self, state, action_mask=None):
        self.q_network.reset_noise()
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get probability distributions
            dist = self.q_network(state_tensor)
            
            # Multiply distributions by the support atoms to get expected Q-values
            q_values = (dist * self.support).sum(2)
            
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(device)
                q_values[0, ~mask_tensor] = -float('inf')
                
        return torch.argmax(q_values).item()
        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        self.q_network.reset_noise()
        #self.target_network.reset_noise()
        
        # Sample with PER
        states, actions, rewards, next_states, dones, next_masks, idxs, weights = self.memory.sample(self.batch_size, self.beta)
        
        # Anneal beta for PER
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        # --- C51 Categorical Projection Math ---
        with torch.no_grad():
            # Double DQN action selection for next state
            next_dist_online = self.q_network(next_states)
            next_q_online = (next_dist_online * self.support).sum(2)
            next_q_online[~next_masks] = -float('inf')
            next_actions = next_q_online.argmax(dim=1)
            
            # Evaluate chosen actions using target network
            next_dist_target = self.target_network(next_states)
            next_dist = next_dist_target[range(self.batch_size), next_actions]
            
            # Compute N-step Bellman targets
            T_z = rewards + (1 - dones) * (self.gamma ** self.n_step) * self.support.unsqueeze(0)
            T_z = T_z.clamp(self.v_min, self.v_max)
            
            b = (T_z - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            weight_l = u.float() - b
            weight_u = b - l.float()

            # If b is exactly an integer, u == l. Force the mass into weight_l
            same_mask = (l == u)
            weight_l[same_mask] = 1.0
            weight_u[same_mask] = 0.0
            
            # Distribute probabilities
            offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).to(device)
            proj_dist = torch.zeros(next_dist.size()).to(device)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        # Current distribution
        dist = self.q_network(states)
        action_dist = dist[range(self.batch_size), actions.squeeze()]
        
        # Cross-Entropy Loss
        elementwise_loss = -(proj_dist * action_dist.clamp(min=1e-5).log()).sum(1)
        loss = (elementwise_loss * weights.squeeze()).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update PER priorities based on the TD-errors
        self.memory.update_priorities(idxs, elementwise_loss.detach().cpu().numpy() + 1e-6)

    def save_checkpoint(self, filepath):
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])