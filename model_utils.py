import copy

import numpy as np
import torch
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm, trange

from evaluation_utils import evaluate_predictions, evaluate_probas


class ModelEngine:
    """
    Creates a model engine in order to rapidly training and evaluating created model.
    This class is compatible with only specific type models.
    """

    def __init__(self, model, criterion, optimizer, isRecurrent=False,
                 scheduler=None, model_name=None, tensorboard_visuals=True, device=None):
        """
        :param model : Created model. Can be ConvNet, CNN_RNN or CRNN type
        :param criterion: Loss function. Should be a loss function compatible with PyTorch
        :param optimizer: Optimizer. Should be compatible with PyTorch.
        :param isRecurrent: Specifies if the model requires hidden inputs like RNN models.
        :param scheduler: Default: None - Learning rate scheduler. Not properly implemented yet.
        :param model_name: Default: None - Desired model name. Can be None or String. If this parameter
        is none, the model will be named according to system current time.
        :param tensorboard_visuals: Default: True - Specifies if tensorboard visuals are required.
        :param device: Default: None - Desired hardware to run model on. If none, best hardware will be
        selected automatically.
        """


        self.scheduler = scheduler
        self.isRecurrent = isRecurrent
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if model_name is None:
            now = datetime.now()
            self.model_name = now.strftime("%d%m%Y%H%M%S")
        else:
            self.model_name = model_name
        self.isTensorboard = tensorboard_visuals

        self.model = self.model.to(device)

    def fit(self, dataloaders, num_epoch=10, inplace=True):
        """
        Initiates training procedure for the current object. Returns the model with the best validation loss.
        :param dataloaders: Dictionary of the dataloaders for both training and validation parts.
        Training dataloader's key should be "train". Validation dataloader's key should be "val".
        :param num_epoch: Number of epochs. Should be integer.
        :param inplace: Default: True - If True, do operation inplace and return None.
        :return: Trained model.
        """

        model = copy.deepcopy(self.model)

        if self.isTensorboard:
            writer = SummaryWriter("runs/" + self.model_name)

        t_epoch = trange(num_epoch, desc="Epochs")
        epoch_loss_prev = 9999999
        best_val = 99999999999999
        for epoch in t_epoch:
            for phase in ['train', 'val']:
                running_loss = 0.0
                if phase == 'train':
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                t_batches = tqdm(dataloaders[phase], desc="Iterations", leave=False, total=len(dataloaders[phase]), )
                for i, (inputs, labels) in enumerate(t_batches):
                    if (i == 0) and self.isRecurrent:
                        hidden_state = model.init_hidden(inputs.size(0))
                    inputs = inputs.to(self.device, dtype=torch.float)
                    labels = labels.to(self.device, dtype=torch.long)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        if self.isRecurrent:
                            outputs, hidden_state = model(inputs, hidden_state)
                        else:
                            outputs = model(inputs)
                        loss = self.criterion(outputs, labels.float())

                        if phase == 'train':
                            loss.backward(retain_graph=True)
                            self.optimizer.step()

                    running_loss += loss.item()

                    if i % 100 == 99 and phase == 'train' and self.isTensorboard:
                        writer.add_scalar('training_loss',
                                          loss.item(),
                                          epoch * len(dataloaders['train']) + i)
                    # t_batch
                    t_batches.set_description("{} loss: {:.4f}".format(phase, loss.item()))
                    t_batches.refresh()

                epoch_loss = running_loss / len(dataloaders[phase])

                if phase == 'train':
                    avg_loss = epoch_loss
                else:
                    val_loss = epoch_loss

                if phase == 'val' and epoch_loss < best_val:
                    best_val = epoch_loss
                    best_model = copy.deepcopy(model)

            t_epoch.set_description("Train Loss: {:.4f} - Val Loss: {:.4f}".format(avg_loss, val_loss))

            if self.isTensorboard:
                writer.add_scalar('val_loss',
                                  val_loss,
                                  epoch)

            if epoch_loss_prev > epoch_loss:
                epoch_loss_prev = epoch_loss
            else:
                print("Loss diddn't decrease. Early stopping.")
                writer.close()
                if inplace:
                    self.model = best_model
                    return
                else:
                    return model

        if self.isTensorboard:
            writer.close()

        if inplace:
            self.model = best_model
        else:
            return best_model

    def predict_proba_with_loader(self, dataloader):
        """
        Makes predictions with the given dataloader.
        :param dataloader: Dataloader with desired data.
        :return: Model outputs.
        """
        self.model.eval()
        predictions = np.empty(0)
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.device, dtype=torch.float)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                predictions = np.append(predictions, outputs.cpu().data.numpy())

        return predictions

    def evaluate_model_with_predictions(self, validation_loader):
        """
        Makes predictions with the given and constructs a confusion matrix.
        :param validation_loader: Dataloader with desired data.
        :return: Confusion matrix
        """

        self.model.eval()
        confusion_matrix = np.zeros((2, 2))
        for i, (inputs, labels) in enumerate(validation_loader):
            inputs = inputs.to(self.device, dtype=torch.float)
            with torch.set_grad_enabled(False):
                if self.isRecurrent:
                    if i == 0:
                        hidden_state = self.model.init_hidden(inputs.size(0))
                    outputs, hidden_state = self.model(inputs, hidden_state)
                else:
                    outputs = self.model(inputs)

                outputs_np = outputs.cpu().data.numpy()
                confusion_matrix += evaluate_predictions(labels.data.numpy(), outputs_np)

        return confusion_matrix / 9

    def evaluate_model_with_probas(self, validation_loader):
        """
        Obtains outputs from the model and returns precisions and recalls for different levels of thresholds.
        :param validation_loader: Dataloader with desired data.
        :return: Precision and recall lists
        """
        self.model.eval()
        output_array = np.empty(0)
        label_array = np.empty(0)
        for i, (inputs, labels) in enumerate(validation_loader):
            # inputs = torch.from_numpy(inputs)
            inputs = inputs.to(self.device, dtype=torch.float)
            with torch.set_grad_enabled(False):
                if self.isRecurrent:
                    if i == 0:
                        hidden_state = self.model.init_hidden(inputs.size(0))
                    outputs, hidden_state = self.model(inputs, hidden_state)
                else:
                    outputs = self.model(inputs)

                outputs_np = outputs.cpu().data.numpy()
                output_array = np.append(output_array, outputs_np)
                label_array = np.append(label_array, labels)

        precisions, recalls = evaluate_probas(label_array, output_array)

        return precisions, recalls