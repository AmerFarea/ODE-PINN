import torch
import torch.optim as optim
from model_IP import SIRPINN
from data_utils_IP import load_and_prepare_sir_data
from training_IP import prepare_dataloader, train_with_physics_loss_sir
from plot_utils import plot_parameters_IP

def main():
    # Define true parameters
    true_params = {
        'beta': 0.5,
        'gamma': 0.05
    }
    # Define device and batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    t_train_set, t_test_set, data_S_train_set, data_I_train_set, data_R_train_set, data_S_test_set, data_I_test_set, data_R_test_set = load_and_prepare_sir_data()
    
    model = SIRPINN(input_dim=1, hidden_dims=[50, 50, 50], output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = prepare_dataloader(t_train_set, data_S_train_set, data_I_train_set, data_R_train_set, batch_size=128, device=device)
    
    epochs = 15000
    patience = 50
    history, train_losses, parameter_estimates = train_with_physics_loss_sir(model, optimizer, epochs, patience, 
                                                                            *next(iter(train_loader)))

    # Plot results
    plot_parameters_IP(parameter_estimates, true_params, epochs)

if __name__ == "__main__":
    main()
