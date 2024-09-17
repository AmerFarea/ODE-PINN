import torch
import torch.optim as optim

def to_tensor(x, device, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad).to(device)

def prepare_dataloader(t_train_set, data_S_train_set, data_I_train_set, data_R_train_set, batch_size, device):
    t_tensor = to_tensor(t_train_set, device, requires_grad=True)
    data_S_tensor = to_tensor(data_S_train_set, device)
    data_I_tensor = to_tensor(data_I_train_set, device)
    data_R_tensor = to_tensor(data_R_train_set, device)
    train_dataset = torch.utils.data.TensorDataset(t_tensor, data_S_tensor, data_I_tensor, data_R_tensor)
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train_with_physics_loss_sir(model, optimizer, epochs, patience, t_tensor, data_S_tensor, data_I_tensor, data_R_tensor):
    train_losses = []
    parameter_estimates = {key: [] for key in model.params.keys()}
    history = {}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(t_tensor)
        S_pred = output[:, 0:1]
        I_pred = output[:, 1:2]
        R_pred = output[:, 2:3]

        beta = model.params['beta']
        gamma = model.params['gamma']

        dS_dt = torch.autograd.grad(outputs=S_pred, inputs=t_tensor, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
        dI_dt = torch.autograd.grad(outputs=I_pred, inputs=t_tensor, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
        dR_dt = torch.autograd.grad(outputs=R_pred, inputs=t_tensor, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]

        physics_loss_S = torch.mean((dS_dt + beta * S_pred * I_pred)**2)
        physics_loss_I = torch.mean((dI_dt - beta * S_pred * I_pred + gamma * I_pred)**2)
        physics_loss_R = torch.mean((dR_dt - gamma * I_pred)**2)
        physics_loss = physics_loss_S + physics_loss_I + physics_loss_R

        data_loss_S = torch.mean((S_pred - data_S_tensor) ** 2)
        data_loss_I = torch.mean((I_pred - data_I_tensor) ** 2)
        data_loss_R = torch.mean((R_pred - data_R_tensor) ** 2)
        data_loss = data_loss_S + data_loss_I + data_loss_R

        loss = 0.3 * physics_loss + 0.7 * data_loss
        loss.backward()
        optimizer.step()

        for key in parameter_estimates:
            parameter_estimates[key].append(model.params[key].item())

        train_losses.append(loss.item())

        UPDATE_INTERVAL = 500
        if epoch % UPDATE_INTERVAL == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
            print(f'Parameters: {", ".join([f"{key}: {model.params[key].item()}" for key in model.params.keys()])}')

            with torch.no_grad():
                output_train_pred = model(t_tensor).detach().cpu().numpy()
                S_train_pred = output_train_pred[:, 0]
                I_train_pred = output_train_pred[:, 1]
                R_train_pred = output_train_pred[:, 2]
            history[epoch] = {
                'S_train_pred': S_train_pred.flatten(),
                'I_train_pred': I_train_pred.flatten(),
                'R_train_pred': R_train_pred.flatten()
            }

        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return history, train_losses, parameter_estimates
