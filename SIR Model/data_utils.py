import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from simulation import sir_model
from scipy.integrate import odeint
import torch

# Simulate data using the SIR model
def simulate_data(beta, gamma, T, num_samples, S0, I0, R0):
    t = np.linspace(0, T, num_samples).reshape(-1, 1)
    y0 = [S0, I0, R0]
    params = (beta, gamma)
    y = odeint(sir_model, y0, t.flatten(), args=params)
    S_data, I_data, R_data = y.T

    data_to_save = pd.DataFrame({'time': t.flatten(), 'Susceptible': S_data, 'Infected': I_data, 'Recovered': R_data})
    data_to_save.to_csv('sir_data.csv', index=False)

    return t, S_data, I_data, R_data

# Load and prepare data
def load_and_prepare_data(test_size=0.2, random_state=42):
    data_loaded = pd.read_csv('sir_data.csv')
    t_loaded = data_loaded['time'].values.reshape(-1, 1)
    data_S_loaded = data_loaded['Susceptible'].values.reshape(-1, 1)
    data_I_loaded = data_loaded['Infected'].values.reshape(-1, 1)
    data_R_loaded = data_loaded['Recovered'].values.reshape(-1, 1)

    t_train_set, t_test_set, data_S_train_set, data_S_test_set, data_I_train_set, data_I_test_set, data_R_train_set, data_R_test_set = train_test_split(
        t_loaded, data_S_loaded, data_I_loaded, data_R_loaded, test_size=test_size, random_state=random_state
    )

    # Sort the data by time for plotting
    t_train_set, data_S_train_set, data_I_train_set, data_R_train_set = zip(*sorted(zip(t_train_set, data_S_train_set, data_I_train_set, data_R_train_set)))
    t_test_set, data_S_test_set, data_I_test_set, data_R_test_set = zip(*sorted(zip(t_test_set, data_S_test_set, data_I_test_set, data_R_test_set)))

    # Convert data to numpy arrays
    t_train_set = np.array(t_train_set)
    t_test_set = np.array(t_test_set)
    data_S_train_set = np.array(data_S_train_set)
    data_I_train_set = np.array(data_I_train_set)
    data_R_train_set = np.array(data_R_train_set)
    data_S_test_set = np.array(data_S_test_set)
    data_I_test_set = np.array(data_I_test_set)
    data_R_test_set = np.array(data_R_test_set)

    return t_train_set, t_test_set, data_S_train_set, data_I_train_set, data_R_train_set, data_S_test_set, data_I_test_set, data_R_test_set


# Convert data to PyTorch tensors
def to_tensor(x, device, requires_grad=False):
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad).to(device)

# Prepare DataLoader for training
def prepare_dataloader(t_train_set, data_S_train_set, data_I_train_set, data_R_train_set, batch_size, device):
    t_tensor = to_tensor(t_train_set, device, requires_grad=True)
    data_S_tensor = to_tensor(data_S_train_set, device)
    data_I_tensor = to_tensor(data_I_train_set, device)
    data_R_tensor = to_tensor(data_R_train_set, device)
    train_dataset = TensorDataset(t_tensor, data_S_tensor, data_I_tensor, data_R_tensor)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
