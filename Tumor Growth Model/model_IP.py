import torch
import torch.nn as nn

class PopulationPINN(nn.Module):
    def __init__(self, input_dim, alpha_init, K_init):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.K = nn.Parameter(torch.tensor(K_init, dtype=torch.float32))

    def forward(self, t):
        t = t.view(-1, 1)
        x = torch.relu(self.fc1(t))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def compute_loss(self, t, population):
        t = t.view(-1, 1).requires_grad_(True)
        pred_P = self.forward(t)
        
        # Clamp predicted values to avoid log(0) issues
        epsilon = 1e-8
        pred_P = torch.clamp(pred_P, min=epsilon)

        # Calculate the derivative of the predicted population
        pred_P.requires_grad_(True)
        d_pred_P_dt = torch.autograd.grad(pred_P, t, torch.ones_like(pred_P), create_graph=True)[0]

        # Logistic growth equation
        alpha = self.alpha
        K = self.K
        f = alpha * torch.log(K / pred_P) * pred_P

        # Physics loss (satisfy the differential equation)
        physics_loss = torch.mean((d_pred_P_dt - f) ** 2)

        # Data loss (fit to the data)
        data_loss = torch.mean((pred_P - population) ** 2)

        # Total loss
        total_loss = physics_loss + data_loss

        return total_loss
