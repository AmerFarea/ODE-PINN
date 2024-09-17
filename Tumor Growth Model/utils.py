import torch
from torch.utils.data import DataLoader, TensorDataset

def to_tensor(x, device, requires_grad=False):
    """Converts a numpy array to a PyTorch tensor."""
    return torch.tensor(x, dtype=torch.float32, requires_grad=requires_grad).to(device)

def prepare_dataloader(t_train_set, data_X_train_set, batch_size, device):
    """Prepares a DataLoader for training."""
    t_tensor = to_tensor(t_train_set, device, requires_grad=True)
    data_X_tensor = to_tensor(data_X_train_set, device)
    train_dataset = TensorDataset(t_tensor, data_X_tensor)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
