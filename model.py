import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class WildfireCNNQNetwork(nn.Module):
    def __init__(self, action_size, grid_size=15, num_channels=11):
        super().__init__()
        
        # Convolutional Feature Extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fixed 64-feature fully connected hidden layer
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        
        self.fc_out = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        return self.fc_out(x)


# Rainbow DQN components
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
    """Categorical Distributional Dueling Network with Noisy Layers."""
    def __init__(self, action_size, grid_size=30, num_channels=11, num_atoms=51, v_min=-300, v_max=300):
        super().__init__()
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Deepened CNN for 30x30 spatial reasoning
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.flat_size = 64 * (grid_size // 2) * (grid_size // 2)
        
        # Dueling Heads with Noisy layers
        self.value_hidden = NoisyLinear(self.flat_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)
        
        self.advantage_hidden = NoisyLinear(self.flat_size, 512)
        self.advantage_out = NoisyLinear(512, action_size * num_atoms)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        v = F.relu(self.value_hidden(x))
        v = self.value_out(v).view(-1, 1, self.num_atoms)
        
        a = F.relu(self.advantage_hidden(x))
        a = self.advantage_out(a).view(-1, self.action_size, self.num_atoms)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q = v + a - a.mean(dim=1, keepdim=True)
        
        # Output probability distribution across the support atoms
        return F.softmax(q, dim=-1)
        
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()