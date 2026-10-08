import torch
import torch.nn as nn
from DQN import DQNAgent

class DDQNAgent(DQNAgent):
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            # Online network selects the best action
            best_next_actions = self.q_network(next_states).argmax(dim=1).unsqueeze(1)
            
            # Target network evaluates that chosen action
            max_next_q_values = self.target_network(next_states).gather(1, best_next_actions)
            
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()