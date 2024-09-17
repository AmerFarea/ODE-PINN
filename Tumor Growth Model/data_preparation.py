import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_prepare_data(output_file='tumor_data.csv'):
    """Loads data from CSV and prepares training and testing sets."""
    data_loaded = pd.read_csv(output_file)
    t_loaded = data_loaded['time'].values.reshape(-1, 1)
    data_X_loaded = data_loaded['solution'].values.reshape(-1, 1)

    t_train_set, t_test_set, data_X_train_set, data_X_test_set = train_test_split(
        t_loaded, data_X_loaded, test_size=0.2, random_state=42
    )

    # Sort data by time
    t_train_set, data_X_train_set = zip(*sorted(zip(t_train_set, data_X_train_set)))
    t_test_set, data_X_test_set = zip(*sorted(zip(t_test_set, data_X_test_set)))

    return np.array(t_train_set), np.array(t_test_set), np.array(data_X_train_set), np.array(data_X_test_set)
