import copy

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from CNN_RNN import CNNwithRNN
from ConvNet import ConvNet
from evaluation_utils import evaluate_model_with_predictions
from train_model import train_model
from DatasetUtils import CNNwithRNNDataset, ConvNetDataset, CRNNDataset


def cross_validation(model,
                     trainset: np.ndarray,
                     fold_number: int):
    model_scores = []
    criterion = nn.BCEWithLogitsLoss()
    fold_generator = sequential_fold_generator(dataset=trainset, fold_number=fold_number)
    if isinstance(model, ConvNet):
        dataset_type = 0
    elif isinstance(model, CNNwithRNN):
        dataset_type = 1
    else:
        dataset_type = 2
    for index, (train, validation) in enumerate(fold_generator):
        if dataset_type == 0:
            train_dataset = ConvNetDataset(train)
            validation_dataset = ConvNetDataset(validation)
            isRecurrent = False
        elif dataset_type == 1:
            train_dataset = CNNwithRNNDataset(train)
            validation_dataset = CNNwithRNNDataset(validation)
            isRecurrent = True
        else:
            train_dataset = CRNNDataset(train)
            validation_dataset = CRNNDataset(validation)
            isRecurrent = True

        dataset_sizes = {
            "train": len(train_dataset),
            "val": len(validation_dataset)
        }
        dataloaders = {
            "train": DataLoader(train_dataset, batch_size=16),
            "val": DataLoader(validation_dataset, batch_size=16)
        }

        model_copy = copy.deepcopy(model)
        optimizer = optim.Adam(model_copy.parameters())

        trained_model = train_model(model_copy, dataloaders, dataset_sizes, criterion, optimizer,
                                    num_epochs=10, isRecurrent=isRecurrent)
        model_scores.append(evaluate_model_with_predictions(trained_model, dataloaders["val"], dataset_sizes["val"]))

    return np.mean(model_scores, axis=0)


def sequential_fold_generator(dataset: np.ndarray,
                              fold_number: int):
    split_rate = len(dataset) // fold_number
    for n in range(fold_number - 1, 0, -1):
        yield np.concatenate([dataset[:split_rate * n], dataset[split_rate * (n + 1):]]),\
              dataset[split_rate * n: split_rate * (n + 1)]


def train_test_split(dataset: np.ndarray, train_set_ratio=.85) -> (np.ndarray, np.ndarray):
    return dataset[:len(dataset) * train_set_ratio], dataset[len(dataset) * train_set_ratio:]



# model = ConvNet()