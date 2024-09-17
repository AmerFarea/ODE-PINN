import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

class SIRPINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SIRPINN, self).__init__()
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        self.params = nn.ParameterDict({
            'beta': nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
            'gamma': nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        })
    
    def forward(self, t):
        return self.network(t)
