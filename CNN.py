import torch
import torch.nn as nn
import torch.nn.functional as F

class WildfireCNNQNetwork(nn.Module):
    def __init__(self, action_size, grid_size=15, num_channels=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_channels = num_channels
        
        # Spatial Stream (CNN)
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size: 64 channels * grid_size * grid_size
        self.flat_size = 64 * grid_size * grid_size
        
        # Fully Connected Value Head
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        # x is expected to be shape: (Batch, Channels, Height, Width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the spatial dimensions
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)