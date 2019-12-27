import torch
from torch import nn
from torch.nn import functional as F


class CNNwithRNN(nn.Module):
    """
    Model for spatio-temporal prediction. The model is as: Input -> CONVNET -> GRU -> Fully Connected -> Output
    """

    def __init__(self, drop_prob=.2, hidden_dim=256):
        """
        Initializes the model.
        :param drop_prob: Dropout probability for GRU layers.
        :param hidden_dim: Hidden dim size for GRU layers.
        """
        super(CNNwithRNN, self).__init__()
        # Calculation of the output size to be added
        self.rnn_layers = 2
        self.hidden_dim = hidden_dim
        self.upsampling1 = self._conv_transpose(2, in_channels=1, out_channels=8, kernel_size=(3, 3))
        self.upsampling2 = self._conv_transpose(2, in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.conv1 = self._conv(2, in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv2 = self._conv(2, in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.rnn = nn.GRU(64, hidden_dim, self.rnn_layers, batch_first=True, dropout=drop_prob)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, inputs, hidden):
        """
        Forward pass for the model
        :param inputs: 1x3x3 sized input.
        :param hidden: Hidden input obtained from earlier time steps
        :return: Output for prediction and hidden output for next time steps.
        """
        x = F.relu(self.upsampling1(inputs))
        x = F.relu(self.upsampling2(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Residual connection
        x += inputs  # b, c, w, h
        x = x.view(x.size(0), x.size(1), -1)  # flattening width and height to preserve spatial information in RNN
        x = x.permute(0, 2, 1)  # permuting tensor according to batch_first = False fashion (b, h-w, c)
        hidden = hidden[:, -inputs.size(0):, :].contiguous()
        x, hidden = self.rnn(x, hidden)
        x = self.dense(x)
        return x, hidden

    def init_hidden(self, batch_size):
        """
        Initializes weights for hidden inputs for t = 0
        :param batch_size: Batch size of the inputs
        :return: Zero tensor for hidden inputs for t = 0
        """
        weights = next(self.parameters()).data
        return weights.new_zeros((self.rnn_layers, batch_size, self.hidden_dim))

    @staticmethod
    def _conv(dim, **kwargs):
        """
        Lazy way to create CNN layers.
        :param dim: Dimension of CNN layers.
        :param kwargs: PyTorch's conv layer attributes
        :return: Convolution layer
        """
        if dim == 1:
            layer = nn.Conv1d(**kwargs)
        elif dim == 2:
            layer = nn.Conv2d(**kwargs)
        elif dim == 3:
            layer = nn.Conv3d(**kwargs)
        else:
            raise NotImplementedError()

        return layer

    @staticmethod
    def _conv_transpose(dim, **kwargs):
        """
        Lazy way to create Transposed CNN layers used for upsampling.
        :param dim: Dimension of transposed CNN layers.
        :param kwargs: PyTorch's transposed conv layer attributes
        :return: Transposed Convolution layer
        """
        if dim == 1:
            layer = nn.ConvTranspose1d(**kwargs)
        elif dim == 2:
            layer = nn.ConvTranspose2d(**kwargs)
        elif dim == 3:
            layer = nn.ConvTranspose3d(**kwargs)
        else:
            raise NotImplementedError()

        return layer

# network = CNNwithRNN()
# out = network(torch.zeros((1, 1, 3, 3)))
# out.shape
