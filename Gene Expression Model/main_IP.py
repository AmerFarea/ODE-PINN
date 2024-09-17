import torch
import torch.optim as optim
from model import PINN
import torch.nn as nn
from data_utils import simulate_data, load_and_prepare_data
from train import prepare_dataloader, train_with_physics_loss,to_tensor
# from plot_utils import plot_training_results, plot_results
from parameter_estimation import train_with_physics_loss_IP, PINN_IP
from GE_plot_utils import *

import warnings
warnings.filterwarnings("ignore")

def main():
# Define true parameters
    true_params = {
        'k_m': 0.6,
        'gamma_m': 0.4,
        'k_p': 0.9,
        'gamma_p': 0.5
    }

        

    # Load and prepare data
    t_train_set, t_test_set, data_m_train_set, data_p_train_set, data_m_test_set, data_p_test_set = load_and_prepare_data()

    # Define device and batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32

    # Prepare DataLoader
    train_loader = prepare_dataloader(t_train_set, data_m_train_set, data_p_train_set, batch_size, device)

    # Define the architecture and training parameters
    input_dim = 1
    hidden_dims = [100, 100]  # Number of neurons in hidden layers
    output_dim = 2  # Two outputs: mRNA and protein
    patience = 1500  # Number of epochs to wait for improvement
    epochs = 20000  # Number of training epochs
    lr = 0.001  # Learning rate for the optimizer


    # Instantiate the model and optimizer
    model_with_physics = PINN_IP(input_dim, hidden_dims, output_dim).to(device)
    optimizer_with_physics = optim.Adam(model_with_physics.parameters(), lr=lr)

    # Convert data to tensors
    t_train_tensor = to_tensor(t_train_set, device, requires_grad=True)
    data_m_train_tensor = to_tensor(data_m_train_set, device)
    data_p_train_tensor = to_tensor(data_p_train_set, device)

    # Train the model and get the training results
    history_with_physics, train_losses_with_physics, parameter_estimates = train_with_physics_loss_IP(
        model_with_physics, optimizer_with_physics, epochs, patience,
        t_train_tensor, data_m_train_tensor, data_p_train_tensor
    )

    

    # Plot parameter estimates and true values
    plot_parameters_IP(parameter_estimates, true_params, epochs)


if __name__ == "__main__":
    main()

