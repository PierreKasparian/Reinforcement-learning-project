import numpy as np
import random
import torch
import torch.nn as nn
from DQN import DQNAgent
from buffer import device

class DDQNAgent(DQNAgent):
    def __init__(self, action_size, grid_size=15, num_channels=11):
        super().__init__(action_size, grid_size, num_channels)
        
        # Override hyperparameters specifically for DDQN stability
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=5e-5)
        self.epsilon_decay = 0.998
        self.tau = 0.005

    def act(self, state, action_mask=None):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            
            if action_mask is not None:
                mask_tensor = torch.tensor(action_mask, dtype=torch.bool).to(device)
                q_values[0, ~mask_tensor] = -float('inf')
            else:
                mask_tensor = torch.ones(self.action_size, dtype=torch.bool, device=device)
        
        # Softmax / Boltzmann exploration targeting promising masked actions
        if random.random() < self.epsilon:
            valid_indices = torch.where(mask_tensor)[0]
            if len(valid_indices) > 0:
                temperature = max(0.1, self.epsilon)
                masked_q = q_values[0, valid_indices] / temperature
                probs = torch.softmax(masked_q, dim=-1).cpu().numpy()
                chosen_idx = np.random.choice(valid_indices.cpu().numpy(), p=probs)
                return int(chosen_idx)
            return random.randint(0, self.action_size - 1)
            
        return torch.argmax(q_values).item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)
        
        q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            # Online network selects the best action
            next_q_values_online = self.q_network(next_states)
            next_q_values_online = next_q_values_online.masked_fill(~next_masks, -float('inf'))  # Mask invalid actions
            best_next_actions = next_q_values_online.argmax(dim=1).unsqueeze(1)
            
            # Target network evaluates that chosen action
            max_next_q_values = self.target_network(next_states).gather(1, best_next_actions)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)