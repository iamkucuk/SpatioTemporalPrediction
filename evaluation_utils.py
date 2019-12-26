import numpy as np
import torch


def model_predict(model, dataloader, device=None):
    """

    :param model:
    :param dataloader:
    :param dataset_size:
    :param device:
    :return:
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    predictions = []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            predictions.append(outputs)

    return outputs


def evaluate_model_with_predictions(model, validation_loader, dataset_size, device=None):
    """

    :param model:
    :param validation_loader:
    :param dataset_size:
    :param device:
    :return:
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    confusion_matrix = np.zeros((2, 2))
    for i, (inputs, labels) in enumerate(validation_loader):
        inputs = inputs.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs_np = outputs.cpu().data.numpy()
            outputs_np[outputs_np > 0] = 1
            outputs_np[outputs_np < 0] = 0
            confusion_matrix += evaluate_predictions(labels.data.numpy(), outputs_np)

    return confusion_matrix / 9


def evaluate_probas(y_true: np.ndarray, y_proba: np.ndarray):
    """

    :param y_true:
    :param y_proba:
    :return:
    """
    pass


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return np.array([[tn, fn], [fp, tp]])
