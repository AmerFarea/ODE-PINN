import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn):
        super(PINN, self).__init__()
        layers = []
        in_dim = input_dim
        
        # Create hidden layers with specified activation function
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Apply Glorot uniform initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, t):
        return self.network(t)
