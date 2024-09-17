import torch
import torch.nn as nn

class PINN(nn.Module):
    """Defines the Physics-Informed Neural Network (PINN)."""
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn):
        super(PINN, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)

def physics_loss_fn(dPdt, P_pred, params):
    """Calculates the physics-informed loss."""
    alpha, K = params
    P_pred = P_pred.squeeze()
    dPdt = dPdt.squeeze()
    P_pred = torch.clamp(P_pred, min=1e-6)  # Prevent log(0)
    dPdt_exact = alpha * torch.log(K / P_pred) * P_pred
    return torch.mean((dPdt - dPdt_exact) ** 2)
