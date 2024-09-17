import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_sir_data(test_size=0.2, random_state=42):
    data_loaded = pd.read_csv('sir_data.csv')
    t_loaded = data_loaded['time'].values.reshape(-1, 1)
    data_S_loaded = data_loaded['Susceptible'].values.reshape(-1, 1)
    data_I_loaded = data_loaded['Infected'].values.reshape(-1, 1)
    data_R_loaded = data_loaded['Recovered'].values.reshape(-1, 1)

    t_train_set, t_test_set, data_S_train_set, data_S_test_set, data_I_train_set, data_I_test_set, data_R_train_set, data_R_test_set = train_test_split(
        t_loaded, data_S_loaded, data_I_loaded, data_R_loaded, test_size=test_size, random_state=random_state
    )

    t_train_set, data_S_train_set, data_I_train_set, data_R_train_set = zip(*sorted(zip(t_train_set, data_S_train_set, data_I_train_set, data_R_train_set)))
    t_test_set, data_S_test_set, data_I_test_set, data_R_test_set = zip(*sorted(zip(t_test_set, data_S_test_set, data_I_test_set, data_R_test_set)))

    return (np.array(t_train_set), np.array(t_test_set), np.array(data_S_train_set), 
            np.array(data_I_train_set), np.array(data_R_train_set), 
            np.array(data_S_test_set), np.array(data_I_test_set), 
            np.array(data_R_test_set))
