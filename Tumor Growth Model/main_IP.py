import torch
from utils_IP import load_data
from model_IP import PopulationPINN
from training_IP import train_model
from TG_plot_utils import *
import numpy as np


import warnings
warnings.filterwarnings("ignore")


def main():
    # Configuration Section
    INPUT_DIM = 1
    LR = 0.001  # Learning rate
    EPOCHS = 1500
    EPOCH_INTERVAL = 500  # Interval for plotting and printing
    PATIENCE = 200  # Patience for early stopping

    filename = "tumor_data.csv"
    timepoints, population = load_data(filename)

    true_alpha = 0.3
    true_K = 100

    # Initialize the model with estimated parameters
    params0 = np.array([1.0, 10.0])
    alpha_init, K_init = params0[0], params0[1]

    # Create and train the PINN model with initial guesses
    model = PopulationPINN(input_dim=INPUT_DIM, alpha_init=alpha_init, K_init=K_init)
    model, alpha_history, K_history, loss_history = train_model(model, timepoints, population, num_epochs=EPOCHS, EPOCH_INTERVAL=EPOCH_INTERVAL, lr=LR)

    # Generate predictions with the trained PINN
    t_dense = np.linspace(timepoints[0], timepoints[-1], num=1000)
    t_dense_tensor = torch.tensor(t_dense, dtype=torch.float32).view(-1, 1)
    solution = model.forward(t_dense_tensor).detach().numpy()

    # Plot results and parameter evolution
    plot_parameters(alpha_history, K_history, true_alpha, true_K)

if __name__ == "__main__":
    main()
