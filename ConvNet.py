import torch
from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    """
    3D Convolutional Neural Network implementation that captures spatial and temporal patterns.
    """

    def __init__(self):
        """
        Initialization function
        """
        super(ConvNet, self).__init__()
        # Calculation of the output size to be added
        self.upsampling1 = self._conv_transpose(3, in_channels=1, out_channels=8, kernel_size=(3, 3, 3))
        self.upsampling2 = self._conv_transpose(3, in_channels=8, out_channels=16, kernel_size=(3, 3, 3))
        self.conv1 = self._conv(3, in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        self.conv2 = self._conv(3, in_channels=32, out_channels=64, kernel_size=(3, 3, 3))
        self.dense = nn.Linear(in_features=64 * 10 * 3 * 3, out_features=9)

    def forward(self, inputs):
        """
        Forward pass of the model
        :param inputs: Input tensor
        :return: Output tensor
        """
        x = F.relu(self.upsampling1(inputs))
        x = F.relu(self.upsampling2(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Residual connection
        x += inputs
        out = self.dense(x.view(x.size(0), -1))
        return out

    @staticmethod
    def _conv(dim, **kwargs):
        """
        Lazy method for creating convolutional layers with desired dimensionality.
        :param dim: Dimension of the convolutional layer
        :param kwargs: Standard PyTorch layer arguments.
        :return: Generated layer
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
        Lazy method for creating transposed convolutional layers with desired dimensionality.
        :param dim: Dimension of the transposed convolutional layer
        :param kwargs: Standard PyTorch layer arguments.
        :return: Generated layer
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


# network = ConvNet()
# out = network(torch.zeros((1, 1, 3, 3, 3)))
# out.shape
