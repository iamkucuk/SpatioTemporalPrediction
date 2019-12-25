import numpy as np
import train_model


def cross_validation(model,
                     trainset: np.ndarray,
                     fold_number: int):

    for X, y in sequential_fold_generator(dataset=trainset, fold_number=fold_number):
        train_model()

    # get fold
    # train net
    # validation evaluation
    # get mean and return for model
    pass


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
    pass
