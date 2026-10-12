import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WildfireCNNQNetwork(nn.Module):
    def __init__(self, action_size, grid_size=30, num_spatial=5, num_global=6):
        super().__init__()
        self.grid_size = grid_size
        
        # Spatial Feature Extractor
        self.conv1 = nn.Conv2d(num_spatial, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Correct math for PyTorch Conv2d with stride=2, pad=1, kernel=3
        conv_out_size = (grid_size + 1) // 2 
        self.flat_size = 64 * conv_out_size * conv_out_size
        
        # Global Feature Extractor
        self.global_fc = nn.Linear(num_global, 64)
        
        # Shared fully connected layer fusing spatial and global
        self.fc_shared = nn.Linear(self.flat_size + 64, 512)
        
        # Branching Heads
        self.head_type = nn.Linear(512, 4)  # NOOP, WATER_DROP, FIREBREAK, TRUCK
        self.head_ori = nn.Linear(512, 2)   # HORIZONTAL, VERTICAL
        self.head_row = nn.Linear(512, grid_size)
        self.head_col = nn.Linear(512, grid_size)

    def forward(self, spatial, global_feats):
        batch_size = spatial.size(0)
        
        # Process Spatial Image
        x = F.relu(self.conv1(spatial))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        # Process Global Vector
        g = F.relu(self.global_fc(global_feats))
        
        # Fuse
        fused = torch.cat([x, g], dim=1)
        fused = F.relu(self.fc_shared(fused))
        
        # Compute independent Q-values for each branch
        q_type = self.head_type(fused)
        q_ori = self.head_ori(fused)
        q_row = self.head_row(fused)
        q_col = self.head_col(fused)
        
        # Spatial Q-values combined -> Shape: (B, G, G)
        q_sp = q_row.unsqueeze(2) + q_col.unsqueeze(1)
        q_sp_flat = q_sp.view(batch_size, -1)
        
        q_noop = q_type[:, 0:1]
        q_wd_h = q_type[:, 1:2] + q_ori[:, 0:1] + q_sp_flat
        q_wd_v = q_type[:, 1:2] + q_ori[:, 1:2] + q_sp_flat
        q_fb_h = q_type[:, 2:3] + q_ori[:, 0:1] + q_sp_flat
        q_fb_v = q_type[:, 2:3] + q_ori[:, 1:2] + q_sp_flat
        q_ft   = q_type[:, 3:4] + q_sp_flat
        
        q_flat = torch.cat([q_noop, q_wd_h, q_wd_v, q_fb_h, q_fb_v, q_ft], dim=1)
        return q_flat


class NoisyLinear(nn.Module):
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
    def __init__(self, action_size, grid_size=30, num_spatial=5, num_global=6, num_atoms=51, v_min=-10, v_max=2):
        super().__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.grid_size = grid_size
        
        self.conv1 = nn.Conv2d(num_spatial, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        conv_out_size = (grid_size + 1) // 2 
        self.flat_size = 64 * conv_out_size * conv_out_size
        
        self.global_fc = nn.Linear(num_global, 64)
        
        # Value Stream
        self.value_hidden = NoisyLinear(self.flat_size + 64, 512)
        self.value_out = NoisyLinear(512, num_atoms)
        
        # Advantage Stream (Branched)
        self.adv_hidden = NoisyLinear(self.flat_size + 64, 512)
        self.adv_type = NoisyLinear(512, 4 * num_atoms)
        self.adv_ori = NoisyLinear(512, 2 * num_atoms)
        self.adv_row = NoisyLinear(512, grid_size * num_atoms)
        self.adv_col = NoisyLinear(512, grid_size * num_atoms)
        
    def forward(self, spatial, global_feats):
        batch_size = spatial.size(0)
        
        x = F.relu(self.conv1(spatial))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)
        
        g = F.relu(self.global_fc(global_feats))
        fused = torch.cat([x, g], dim=1)
        
        # Value computation
        v = F.relu(self.value_hidden(fused))
        v = self.value_out(v).view(batch_size, 1, self.num_atoms)
        
        # Branched Advantage computations
        a_hidden = F.relu(self.adv_hidden(fused))
        a_type = self.adv_type(a_hidden).view(batch_size, 4, self.num_atoms)
        a_ori = self.adv_ori(a_hidden).view(batch_size, 2, self.num_atoms)
        a_row = self.adv_row(a_hidden).view(batch_size, self.grid_size, self.num_atoms)
        a_col = self.adv_col(a_hidden).view(batch_size, self.grid_size, self.num_atoms)
        
        a_sp = a_row.unsqueeze(2) + a_col.unsqueeze(1)
        a_sp_flat = a_sp.view(batch_size, self.grid_size * self.grid_size, self.num_atoms)
        
        a_noop = a_type[:, 0:1, :]
        a_wd_h = a_type[:, 1:2, :] + a_ori[:, 0:1, :] + a_sp_flat
        a_wd_v = a_type[:, 1:2, :] + a_ori[:, 1:2, :] + a_sp_flat
        a_fb_h = a_type[:, 2:3, :] + a_ori[:, 0:1, :] + a_sp_flat
        a_fb_v = a_type[:, 2:3, :] + a_ori[:, 1:2, :] + a_sp_flat
        a_ft   = a_type[:, 3:4, :] + a_sp_flat
        
        a_flat = torch.cat([a_noop, a_wd_h, a_wd_v, a_fb_h, a_fb_v, a_ft], dim=1)
        q = v + a_flat - a_flat.mean(dim=1, keepdim=True)
        return F.softmax(q, dim=-1)
        
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_type.reset_noise()
        self.adv_ori.reset_noise()
        self.adv_row.reset_noise()
        self.adv_col.reset_noise()