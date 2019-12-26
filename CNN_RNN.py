import torch
from torch import nn
from torch.nn import functional as F


class CNNwithRNN(nn.Module):
    """

    """

    def __init__(self, drop_prob=.2, hidden_dim=256):
        super(CNNwithRNN, self).__init__()
        # Calculation of the output size to be added
        self.hidden_dim = hidden_dim
        self.upsampling1 = self._conv_transpose(2, in_channels=1, out_channels=8, kernel_size=(3, 3))
        self.upsampling2 = self._conv_transpose(2, in_channels=8, out_channels=16, kernel_size=(3, 3))
        self.conv1 = self._conv(2, in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv2 = self._conv(2, in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.rnn = nn.GRU(64, hidden_dim, 2, batch_first=True, dropout=drop_prob)
        self.dense = nn.Linear(hidden_dim, 9)

    def forward(self, inputs, hidden=None):
        x = F.relu(self.upsampling1(inputs))
        x = F.relu(self.upsampling2(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Residual connection
        x += inputs  # b, c, w, h
        x = x.view(x.size(0), x.size(1), -1)  # flattening width and height to preserve spatial information in RNN
        x = x.permute(0, 2, 1)  # permuting tensor according to batch_first = False fashion (h-w, b, c)
        x, hidden = self.rnn(x, hidden)
        x = self.dense(F.relu(x[:, -1]))
        return x, hidden

    @staticmethod
    def _conv(dim, **kwargs):
        """

        :param dim:
        :param kwargs:
        :return:
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

        :param dim:
        :param kwargs:
        :return:
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
