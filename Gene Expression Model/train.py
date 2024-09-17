import torch
import torch.optim as optim


# Convert data to PyTorch tensors
def to_tensor(x, device, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad).to(device)

def prepare_dataloader(t_train_set, data_m_train_set, data_p_train_set, batch_size, device):
    from torch.utils.data import DataLoader, TensorDataset
    t_tensor = to_tensor(t_train_set, device, requires_grad=True)
    data_m_tensor = to_tensor(data_m_train_set, device)
    data_p_tensor = to_tensor(data_p_train_set, device)
    train_dataset = TensorDataset(t_tensor, data_m_tensor, data_p_tensor)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train_with_physics_loss(model, optimizer, epochs, patience, t_tensor, data_m_tensor, data_p_tensor, t_test_tensor, data_m_test_tensor, data_p_test_tensor, k_m, gamma_m, k_p, gamma_p, PN):
    train_losses = []
    test_losses = []
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

        # Store train loss
        train_losses.append(loss.item())

        # Compute and store test loss
        model.eval()
        with torch.no_grad():
            output_test = model(t_test_tensor)
            m_test_pred = output_test[:, 0:1]
            p_test_pred = output_test[:, 1:2]

            test_loss_m = torch.mean((m_test_pred - data_m_test_tensor) ** 2)
            test_loss_p = torch.mean((p_test_pred - data_p_test_tensor) ** 2)
            test_loss = 0.3 * physics_loss_m + 0.7 * test_loss_m + 0.7 * test_loss_p
            test_losses.append(test_loss.item())

            # Save predictions every 500 epochs
            if epoch % 500 == 0:
                output_train_pred = model(t_tensor).detach().cpu().numpy()
                m_train_pred = output_train_pred[:, 0]
                p_train_pred = output_train_pred[:, 1]
                history[epoch] = {
                    'm_train_pred': m_train_pred.flatten(),
                    'p_train_pred': p_train_pred.flatten(),
                    'm_test_pred': m_test_pred.cpu().numpy().flatten(),
                    'p_test_pred': p_test_pred.cpu().numpy().flatten()
                }

        # Print progress every 500 epochs
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')

        # Implement early stopping based on validation loss
        if loss.item() < best_val_loss:
            best_val_loss = loss.item()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return history, train_losses, test_losses
