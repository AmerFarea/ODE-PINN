import torch
from ode_solver_IP import optimize_parameters

def train_model(model, timepoints, population, num_epochs, EPOCH_INTERVAL, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    alpha_history = []
    K_history = []

    # Normalize data for better training
    timepoints_tensor = torch.tensor(timepoints, dtype=torch.float32).view(-1, 1)
    population_tensor = torch.tensor(population, dtype=torch.float32).view(-1, 1)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = model.compute_loss(timepoints_tensor, population_tensor)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # Record the parameter values
        alpha_history.append(model.alpha.item())
        K_history.append(model.K.item())

        if (epoch + 1) % EPOCH_INTERVAL == 0:
            # Optimize parameters periodically
            alpha_init, K_init = optimize_parameters(timepoints, population)
            model.alpha.data = torch.tensor(alpha_init, dtype=torch.float32)
            model.K.data = torch.tensor(K_init, dtype=torch.float32)

        if (epoch + 1) % EPOCH_INTERVAL == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_value:.4f}, alpha: {model.alpha.item():.4f}, K: {model.K.item():.4f}")

    return model, alpha_history, K_history, loss_history
