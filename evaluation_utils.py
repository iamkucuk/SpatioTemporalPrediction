import torch


def model_predict(model, dataloader, dataset_size, device=None):
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
        labels = labels.to(device, dtype=torch.float)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            predictions.append(outputs)


    return outputs
