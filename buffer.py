import torch
import numpy as np
import random
import pickle
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic replay buffer for DQN and DDQN
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_masks), dtype=torch.bool).to(device)
        )
        
    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.buffer = pickle.load(f)

# Rainbow DQN components
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
        # N-Step parameters
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.n_step_buffer.append((state, action, reward, next_state, done, next_mask))
        
        if len(self.n_step_buffer) == self.n_step:
            s, a, R, n_s, d, n_m = self._get_n_step_info()
            self.tree.add(self.max_priority ** self.alpha, (s, a, R, n_s, d, n_m))
            
        if done:
            while len(self.n_step_buffer) > 0:
                s, a, R, n_s, d, n_m = self._get_n_step_info()
                self.tree.add(self.max_priority ** self.alpha, (s, a, R, n_s, d, n_m))
                self.n_step_buffer.popleft()

    def _get_n_step_info(self):
        R = 0
        for idx, transition in enumerate(self.n_step_buffer):
            R += transition[2] * (self.gamma ** idx)
            if transition[4]: # If done
                break
        s, a = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        n_s, d, n_m = transition[3], transition[4], transition[5]
        return s, a, R, n_s, d, n_m

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.tree[0] / batch_size
        priorities = []

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.tree[0]
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device),
            torch.tensor(np.array(next_masks), dtype=torch.bool).to(device),
            idxs,
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)
        )

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            self.max_priority = max(self.max_priority, float(p))
            self.tree.update(idx, p ** self.alpha)

    def __len__(self):
        return self.tree.n_entries