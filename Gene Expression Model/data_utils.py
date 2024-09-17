import numpy as np
import pandas as pd
import torch
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split




# Simulate data using the provided ODEs
def simulate_data(PN, k_m, gamma_m, k_p, gamma_p, T, num_samples, m0, p0):
    def system(vars, t, params):
        m, p = vars
        k_m, gamma_m, k_p, gamma_p, PN = params
        dm_dt = k_m * PN - gamma_m * m
        dp_dt = k_p * m - gamma_p * p
        return [dm_dt, dp_dt]

    t = np.linspace(0, T, num_samples).reshape(-1, 1)
    params = [k_m, gamma_m, k_p, gamma_p, PN]
    y = odeint(system, [m0, p0], t.flatten(), args=(params,))
    m_data, p_data = y.T

    data_to_save = pd.DataFrame({'time': t.flatten(), 'mRNA': m_data, 'protein': p_data})
    data_to_save.to_csv('gene_data.csv', index=False)

    return t, m_data, p_data

# Load and prepare data
def load_and_prepare_data(test_size=0.2, random_state=42):
    data_loaded = pd.read_csv('gene_data.csv')
    t_loaded = data_loaded['time'].values.reshape(-1, 1)
    data_m_loaded = data_loaded['mRNA'].values.reshape(-1, 1)
    data_p_loaded = data_loaded['protein'].values.reshape(-1, 1)

    t_train_set, t_test_set, data_m_train_set, data_m_test_set, data_p_train_set, data_p_test_set = train_test_split(
        t_loaded, data_m_loaded, data_p_loaded, test_size=test_size, random_state=random_state
    )

    # Sort the data by time for plotting
    t_train_set, data_m_train_set, data_p_train_set = zip(*sorted(zip(t_train_set, data_m_train_set, data_p_train_set)))
    t_test_set, data_m_test_set, data_p_test_set = zip(*sorted(zip(t_test_set, data_m_test_set, data_p_test_set)))

    # Convert data to numpy arrays
    t_train_set = np.array(t_train_set)
    t_test_set = np.array(t_test_set)
    data_m_train_set = np.array(data_m_train_set)
    data_p_train_set = np.array(data_p_train_set)
    data_m_test_set = np.array(data_m_test_set)
    data_p_test_set = np.array(data_p_test_set)

    return t_train_set, t_test_set, data_m_train_set, data_p_train_set, data_m_test_set, data_p_test_set

