import torch
import torch.nn as nn
from DQN import DQNAgent

class DDQNAgent(DQNAgent):
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
            next_q_values_online = self.q_network(next_sp_states, next_gl_states)
            next_q_values_online[~next_masks] = -1e8 
            
            best_next_actions = next_q_values_online.argmax(dim=1).unsqueeze(1)
            
            max_next_q_values = self.target_network(next_sp_states, next_gl_states).gather(1, best_next_actions)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
            
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.total_steps % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())