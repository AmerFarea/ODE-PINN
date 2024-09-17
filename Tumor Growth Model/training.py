import torch
import torch.optim as optim

def train_with_physics_loss(model, optimizer, epochs, patience, t_tensor, data_X_tensor, t_test_tensor, data_X_test_tensor, physics_loss_fn, params):
    """Trains the PINN model with a physics-informed loss function."""
    train_losses = []
    val_losses = []
    history = {}
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass through the model
        P_pred = model(t_tensor)

        # Compute dP/dt using autograd
        dPdt = torch.autograd.grad(outputs=P_pred, inputs=t_tensor,
                                   grad_outputs=torch.ones_like(P_pred),
                                   create_graph=True)[0]

        # Compute the physics loss
        physics_loss = physics_loss_fn(dPdt, P_pred, params)

        # Compute the data loss
        data_loss = torch.mean((P_pred - data_X_tensor) ** 2)

        # Total loss is a weighted sum of physics loss and data loss
        loss = 0.3 * physics_loss + 0.7 * data_loss
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Compute validation loss
        model.eval()
        with torch.no_grad():
            P_test_pred = model(t_test_tensor)
            val_loss = torch.mean((P_test_pred - data_X_test_tensor) ** 2).item()
            val_losses.append(val_loss)

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}, Validation Loss: {val_loss}')

            with torch.no_grad():
                P_train_pred = model(t_tensor).detach().cpu().numpy()
                P_test_pred = model(t_test_tensor).detach().cpu().numpy()
            history[epoch] = {
                'train_pred': P_train_pred,
                'test_pred': P_test_pred
            }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    return history, train_losses, val_losses
