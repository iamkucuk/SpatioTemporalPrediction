import numpy as np
import torch
import matplotlib.pyplot as plt


# def model_predict(model, dataloader, device=None):
#     """
#
#     :param model:
#     :param dataloader:
#     :param dataset_size:
#     :param device:
#     :return:
#     """
#
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model.eval()
#
#     predictions = []
#
#     for i, (inputs, labels) in enumerate(dataloader):
#         inputs = inputs.to(device, dtype=torch.float)
#
#         with torch.set_grad_enabled(False):
#             outputs = model(inputs)
#
#             predictions.append(outputs)
#
#     return outputs


# def evaluate_model_with_predictions(model, validation_loader, isRecurrent=False, device=None):
#     """
#
#     :param isRecurrent:
#     :param model:
#     :param validation_loader:
#     :param dataset_size:
#     :param device:
#     :return:
#     """
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model.eval()
#     confusion_matrix = np.zeros((2, 2))
#     for i, (inputs, labels) in enumerate(validation_loader):
#         inputs = inputs.to(device, dtype=torch.float)
#         with torch.set_grad_enabled(False):
#             if isRecurrent:
#                 if i == 0:
#                     hidden_state = model.init_hidden(inputs.size(0))
#                 outputs, hidden_state = model(inputs, hidden_state)
#             else:
#                 outputs = model(inputs)
#
#             outputs_np = outputs.cpu().data.numpy()
#             confusion_matrix += evaluate_predictions(labels.data.numpy(), outputs_np)
#
#     return confusion_matrix / 9


# def evaluate_model_with_probas(model, validation_loader, isRecurrent=True, device=None):
#     if device is None:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     model.eval()
#     output_array = np.empty(0)
#     label_array = np.empty(0)
#     for i, (inputs, labels) in enumerate(validation_loader):
#         # inputs = torch.from_numpy(inputs)
#         inputs = inputs.to(device, dtype=torch.float)
#         with torch.set_grad_enabled(False):
#             if isRecurrent:
#                 if i == 0:
#                     hidden_state = model.init_hidden(inputs.size(0))
#                 outputs, hidden_state = model(inputs, hidden_state)
#             else:
#                 outputs = model(inputs)
#
#             outputs_np = outputs.cpu().data.numpy()
#             output_array = np.append(output_array, outputs_np)
#             label_array = np.append(label_array, labels)
#
#     precisions, recalls = evaluate_probas(label_array, output_array)
#
#     return precisions, recalls


def evaluate_probas(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Evaluation metric (Precision-Recall Curve) for probability evaluation
    :param y_true: Ground truth
    :param y_proba: Output of model. If the model output is not between 0 and 1, a sigmoid function will be applied.
    :return: Precision and recall lists
    """
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    if (np.min(y_proba) < 0) or (np.max(y_proba) > 1):
        y_proba = sigmoid(y_proba)

    thresholds = np.arange(0.0, 1.0, .01)

    # pos = sum(y_true)
    # neg = len(y_true) - pos

    recalls = []
    precisions = []

    for threshold in thresholds:
        confusion_matrix = evaluate_predictions(y_true=y_true, y_pred=y_proba, threshold=threshold)
        tn, fn = confusion_matrix[0]
        fp, tp = confusion_matrix[1]
        precisions.append(tp / (tp + fp))
        recalls.append(tp / (tp + fn))

    return precisions, recalls


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = .00001) -> np.ndarray:
    """
    Outputs confusion matrix.
    :param threshold: Default .000001 - Threshold be applied if y_pred is not predicted as a class (0 or 1)
    :param y_true: Ground truth
    :param y_pred: Output of model. Can be both prediction or predictid probabiltiy.
    :return: 2x2 Confusion matrix
    """
    y_pred = np.copy(y_pred)
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred < threshold] = 0
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return np.array([[tn, fn], [fp, tp]])
