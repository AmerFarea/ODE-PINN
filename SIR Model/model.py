# Define the Physics-Informed Neural Network (PINN)
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn):
        super(PINN, self).__init__()
        layers = []
        in_dim = input_dim
        
        # Create hidden layers with the selected activation function
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, t):
        return self.network(t)