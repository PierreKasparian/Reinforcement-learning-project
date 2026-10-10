import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WildfireCNNQNetwork(nn.Module):
    def __init__(self, action_size, grid_size=30, num_channels=11):
        super().__init__()
        self.grid_size = grid_size
        
        # Shared CNN Feature Extractor
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.flat_size = 64 * (grid_size // 2) * (grid_size // 2)
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(self.flat_size, 512)
        
        # Branching Heads
        self.head_type = nn.Linear(512, 4)  # NOOP, WATER_DROP, FIREBREAK, TRUCK
        self.head_ori = nn.Linear(512, 2)   # HORIZONTAL, VERTICAL
        self.head_row = nn.Linear(512, grid_size)
        self.head_col = nn.Linear(512, grid_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(batch_size, -1)
        x = F.relu(self.fc_shared(x))
        
        # Compute independent Q-values for each branch
        q_type = self.head_type(x)
        q_ori = self.head_ori(x)
        q_row = self.head_row(x)
        q_col = self.head_col(x)
        
        # Recombine into a flat vector using tensor broadcasting to match actions.py
        # Spatial Q-values combined -> Shape: (B, G, G)
        q_sp = q_row.unsqueeze(2) + q_col.unsqueeze(1)
        # Flatten spatial grid -> Shape: (B, G*G)
        q_sp_flat = q_sp.view(batch_size, -1)
        
        # Map Q-values back to the 5 grid-blocks (+1 NOOP) defined in encode_action/decode_action
        q_noop = q_type[:, 0:1]
        q_wd_h = q_type[:, 1:2] + q_ori[:, 0:1] + q_sp_flat
        q_wd_v = q_type[:, 1:2] + q_ori[:, 1:2] + q_sp_flat
        q_fb_h = q_type[:, 2:3] + q_ori[:, 0:1] + q_sp_flat
        q_fb_v = q_type[:, 2:3] + q_ori[:, 1:2] + q_sp_flat
        q_ft   = q_type[:, 3:4] + q_sp_flat
        
        # Concatenate into the final flat vector of size (1 + 5*G^2)
        q_flat = torch.cat([q_noop, q_wd_h, q_wd_v, q_fb_h, q_fb_v, q_ft], dim=1)
        
        return q_flat


class NoisyLinear(nn.Module):
    """Factorized NoisyLinear layer for state-dependent exploration."""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowCNNQNetwork(nn.Module):
    """Branched Categorical Distributional Dueling Network with Noisy Layers."""
    def __init__(self, action_size, grid_size=30, num_channels=11, num_atoms=51, v_min=-300, v_max=300):
        super().__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.grid_size = grid_size
        
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.flat_size = 64 * (grid_size // 2) * (grid_size // 2)
        
        # Value Stream
        self.value_hidden = NoisyLinear(self.flat_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)
        
        # Advantage Stream (Branched)
        self.adv_hidden = NoisyLinear(self.flat_size, 512)
        self.adv_type = NoisyLinear(512, 4 * num_atoms)
        self.adv_ori = NoisyLinear(512, 2 * num_atoms)
        self.adv_row = NoisyLinear(512, grid_size * num_atoms)
        self.adv_col = NoisyLinear(512, grid_size * num_atoms)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        # Value computation
        v = F.relu(self.value_hidden(x))
        v = self.value_out(v).view(batch_size, 1, self.num_atoms)
        
        # Branched Advantage computations
        a_hidden = F.relu(self.adv_hidden(x))
        a_type = self.adv_type(a_hidden).view(batch_size, 4, self.num_atoms)
        a_ori = self.adv_ori(a_hidden).view(batch_size, 2, self.num_atoms)
        a_row = self.adv_row(a_hidden).view(batch_size, self.grid_size, self.num_atoms)
        a_col = self.adv_col(a_hidden).view(batch_size, self.grid_size, self.num_atoms)
        
        # Recombine spatial advantages -> (B, G, G, num_atoms) -> (B, G*G, num_atoms)
        a_sp = a_row.unsqueeze(2) + a_col.unsqueeze(1)
        a_sp_flat = a_sp.view(batch_size, self.grid_size * self.grid_size, self.num_atoms)
        
        # Map advantages to the 5 grid-blocks (+1 NOOP)
        a_noop = a_type[:, 0:1, :]
        a_wd_h = a_type[:, 1:2, :] + a_ori[:, 0:1, :] + a_sp_flat
        a_wd_v = a_type[:, 1:2, :] + a_ori[:, 1:2, :] + a_sp_flat
        a_fb_h = a_type[:, 2:3, :] + a_ori[:, 0:1, :] + a_sp_flat
        a_fb_v = a_type[:, 2:3, :] + a_ori[:, 1:2, :] + a_sp_flat
        a_ft   = a_type[:, 3:4, :] + a_sp_flat
        
        # Concatenate to form the full flattened advantage atom logits
        a_flat = torch.cat([a_noop, a_wd_h, a_wd_v, a_fb_h, a_fb_v, a_ft], dim=1)
        
        # Dueling Network combination: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q = v + a_flat - a_flat.mean(dim=1, keepdim=True)
        
        # Output probability distribution across the support atoms
        return F.softmax(q, dim=-1)
        
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_type.reset_noise()
        self.adv_ori.reset_noise()
        self.adv_row.reset_noise()
        self.adv_col.reset_noise()