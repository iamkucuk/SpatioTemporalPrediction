import numpy as np
import torch
from torch.utils.data import Dataset


class ConvNetDataset(Dataset):
    """

    :param dataset:
    :param past:
    :param file_path:
    """

    def __init__(self, dataset=None, past=10, file_path="data/train_set.npy"):
        self.past = past
        if dataset is None:
            self.dataset = np.load(file_path)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset) - self.past

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.data.numpy()
        idx = idx + self.past
        features = self.dataset[idx - self.past: idx, :, :]
        target = self.dataset[idx, :, :]
        return np.expand_dims(features, axis=0), target.reshape(-1, 1).squeeze(-1)


class CNNwithRNNDataset(Dataset):
    """

    :param dataset:
    :param file_path:
    """

    def __init__(self, dataset=None, file_path="data/train_set.npy"):
        # self.past = past
        if dataset is None:
            self.dataset = np.load(file_path)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.data.numpy()
        features = self.dataset[idx, :, :]
        target = self.dataset[idx + 1, :, :].reshape(-1, 1)
        return np.expand_dims(features, axis=0), target


class CRNNDataset(Dataset):
    """

    :param dataset:
    :param file_path:
    """

    def __init__(self, dataset=None, file_path="data/train_set.npy"):
        # self.past = past
        if dataset is None:
            self.dataset = np.load(file_path)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset) - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.data.numpy()
        # idx = idx + self.past
        features = self.dataset[idx, :, :]
        target = self.dataset[idx + 1, :, :]
        return np.expand_dims(features, axis=0), target.reshape(-1, 1).squeeze(-1)
