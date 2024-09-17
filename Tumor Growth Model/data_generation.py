import numpy as np
import pandas as pd
from scipy.integrate import odeint

def ode_model(X, t, alpha, K):
    """Defines the differential equation model."""
    dXdt = alpha * np.log(K / X) * X
    return dXdt

def simulate_data(ode_func, params, T, num_samples, y0, output_file='tumor_data.csv'):
    """Simulates data based on the ODE and saves to a CSV file."""
    t = np.linspace(0, T, num_samples).reshape(-1, 1)
    y = odeint(ode_func, y0, t.flatten(), args=params)
    data_X = y.flatten()
    data_to_save = pd.DataFrame({'time': t.flatten(), 'solution': data_X})
    data_to_save.to_csv(output_file, index=False)
    print(f'Saved data to {output_file}')
    return t, data_X
