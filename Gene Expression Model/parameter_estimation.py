import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import xavier_uniform_
# Define the Physics-Informed Neural Network (PINN)
class PINN_IP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PINN_IP, self).__init__()
        layers = []
        in_dim = input_dim
        
        # Create hidden layers with tanh and relu activation
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if i % 2 == 0:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
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
        
        # Parameters to be learned
        self.params = nn.ParameterDict({
            'k_m': nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
            'gamma_m': nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
            'k_p': nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
            'gamma_p': nn.Parameter(torch.tensor(1.0, dtype=torch.float32)),
        })
    
    def forward(self, t):
        return self.network(t)


def train_with_physics_loss_IP(model, optimizer, epochs, patience, t_tensor, data_m_tensor, data_p_tensor):
    train_losses = []
    parameter_estimates = {key: [] for key in model.params.keys()}
    history = {}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(t_tensor)
        m_pred = output[:, 0:1]  # Extract m from the output
        p_pred = output[:, 1:2]  # Extract p from the output

        # Extract parameters
        k_m = model.params['k_m']
        gamma_m = model.params['gamma_m']
        k_p = model.params['k_p']
        gamma_p = model.params['gamma_p']
        PN = 50  # Fixed value of PN

        # Compute dP/dt and dM/dt using autograd
        dm_dt = torch.autograd.grad(outputs=m_pred, inputs=t_tensor,
                                    grad_outputs=torch.ones_like(m_pred),
                                    create_graph=True)[0]
        dp_dt = torch.autograd.grad(outputs=p_pred, inputs=t_tensor,
                                    grad_outputs=torch.ones_like(p_pred),
                                    create_graph=True)[0]

        # Compute the physics loss
        physics_loss_m = torch.mean((dm_dt - k_m * PN + gamma_m * m_pred)**2)
        physics_loss_p = torch.mean((dp_dt - k_p * m_pred + gamma_p * p_pred)**2)
        physics_loss = physics_loss_m + physics_loss_p

        # Compute the data loss
        data_loss_m = torch.mean((m_pred - data_m_tensor) ** 2)
        data_loss_p = torch.mean((p_pred - data_p_tensor) ** 2)
        data_loss = data_loss_m + data_loss_p

        # Total loss is a weighted sum of physics loss and data loss
        loss = 0.3 * physics_loss + 0.7 * data_loss
        loss.backward()
        optimizer.step()

        # Record parameter estimates
        for key in parameter_estimates:
            parameter_estimates[key].append(model.params[key].item())

        train_losses.append(loss.item())

        # Print progress and parameter estimates every UPDATE_INTERVAL epochs
        UPDATE_INTERVAL = 500
        if epoch % UPDATE_INTERVAL == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            print(f'Parameters: {", ".join([f"{key}: {model.params[key].item()}" for key in model.params.keys()])}')

            # Save predictions every UPDATE_INTERVAL epochs
            with torch.no_grad():
                output_train_pred = model(t_tensor).detach().cpu().numpy()
                m_train_pred = output_train_pred[:, 0]
                p_train_pred = output_train_pred[:, 1]
            history[epoch] = {
                'm_train_pred': m_train_pred.flatten(),
                'p_train_pred': p_train_pred.flatten()
            }

        # Implement early stopping based on validation loss
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return history, train_losses, parameter_estimates
