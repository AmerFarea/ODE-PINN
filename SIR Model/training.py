import torch
import torch.optim as optim
import torch.nn as nn
# Train the PINN model with physics loss
def train_with_physics_loss(model, optimizer, epochs, patience, t_train_tensor, data_S_train_tensor, data_I_train_tensor, data_R_train_tensor, 
                            t_val_tensor, data_S_val_tensor, data_I_val_tensor, data_R_val_tensor, beta, gamma):
    train_losses = []
    val_losses = []
    history = {}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass through the model
        output = model(t_train_tensor)
        S_pred = output[:, 0:1]  # Extract S from the output
        I_pred = output[:, 1:2]  # Extract I from the output
        R_pred = output[:, 2:3]  # Extract R from the output

        # Compute dS/dt, dI/dt, and dR/dt using autograd
        dS_dt = torch.autograd.grad(outputs=S_pred, inputs=t_train_tensor,
                                    grad_outputs=torch.ones_like(S_pred),
                                    create_graph=True)[0]
        dI_dt = torch.autograd.grad(outputs=I_pred, inputs=t_train_tensor,
                                    grad_outputs=torch.ones_like(I_pred),
                                    create_graph=True)[0]
        dR_dt = torch.autograd.grad(outputs=R_pred, inputs=t_train_tensor,
                                    grad_outputs=torch.ones_like(R_pred),
                                    create_graph=True)[0]

        # Compute the physics loss
        physics_loss_S = torch.mean((dS_dt + beta * S_pred * I_pred)**2)
        physics_loss_I = torch.mean((dI_dt - beta * S_pred * I_pred + gamma * I_pred)**2)
        physics_loss_R = torch.mean((dR_dt - gamma * I_pred)**2)
        physics_loss = physics_loss_S + physics_loss_I + physics_loss_R

        # Compute the data loss
        data_loss_S = torch.mean((S_pred - data_S_train_tensor) ** 2)
        data_loss_I = torch.mean((I_pred - data_I_train_tensor) ** 2)
        data_loss_R = torch.mean((R_pred - data_R_train_tensor) ** 2)
        data_loss = data_loss_S + data_loss_I + data_loss_R

        # Total loss is a weighted sum of physics loss and data loss
        loss = 0.3 * physics_loss + 0.7 * data_loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Validation loss
        model.eval()
        with torch.no_grad():
            output_val = model(t_val_tensor)
            S_val_pred = output_val[:, 0:1]
            I_val_pred = output_val[:, 1:2]
            R_val_pred = output_val[:, 2:3]

            data_loss_S_val = torch.mean((S_val_pred - data_S_val_tensor) ** 2)
            data_loss_I_val = torch.mean((I_val_pred - data_I_val_tensor) ** 2)
            data_loss_R_val = torch.mean((R_val_pred - data_R_val_tensor) ** 2)
            val_loss = data_loss_S_val + data_loss_I_val + data_loss_R_val
            val_losses.append(val_loss.item())

        # Early stopping based on validation loss
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        # Print progress every 500 epochs
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
            with torch.no_grad():
                output_train_pred = model(t_train_tensor).detach().cpu().numpy()
                S_train_pred = output_train_pred[:, 0]
                I_train_pred = output_train_pred[:, 1]
                R_train_pred = output_train_pred[:, 2]
            history[epoch] = {
                'S_train_pred': S_train_pred, 'I_train_pred': I_train_pred, 'R_train_pred': R_train_pred
            }

    return train_losses, val_losses, history