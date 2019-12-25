import copy

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from train_model import train_model
from ConvNetDataset import ConvNetDataset


def cross_validation(model,
                     trainset: np.ndarray,
                     fold_number: int):
    model_scores = []
    criterion = nn.NLLLoss()
    for train, validation in sequential_fold_generator(dataset=trainset, fold_number=fold_number):
        train_dataset = ConvNetDataset(train)
        validation_dataset = ConvNetDataset(validation)
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

        trained_model = train_model(model_copy, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10)
        model_scores.append(evaluate_predictions(trained_model(validation), validation))

    return np.mean(model_scores)


def sequential_fold_generator(dataset: np.ndarray,
                              fold_number: int):
    split_rate = len(dataset) // fold_number
    n = fold_number - 1
    for n in range(fold_number, 0, -1):
        yield np.concatenate([dataset[:split_rate * n], dataset[split_rate * (n + 1):]]), dataset[
                                                                                          split_rate * n: split_rate * (
                                                                                                  n + 1)]


def train_test_split(dataset: np.ndarray, train_set_ratio=.85) -> (np.ndarray, np.ndarray):
    return dataset[:len(dataset) * train_set_ratio], dataset[len(dataset) * train_set_ratio:]


def evaluate_probas(y_true: np.ndarray, y_proba: np.ndarray):
    pass


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return np.matrix[[tp, fp], [fn, tn]]

# model = ConvNet()