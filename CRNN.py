import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init


class GRU2DCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, device):
        super().__init__()
        self.device = device
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, inputs, prev_state):

        # get batch and spatial sizes
        batch_size = inputs.data.size()[0]
        spatial_size = inputs.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = Variable(torch.zeros(state_size)).to(self.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([inputs, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([inputs, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class CRNN(nn.Module):

    def __init__(self, input_size=1, device="cpu"):
        super(CRNN, self).__init__()
        self.input_size = input_size
        self.layer1 = GRU2DCell(input_size, 32, 3, device)
        self.layer2 = GRU2DCell(32, 64, 3, device)
        self.layer3 = GRU2DCell(64, 16, 3, device)
        self.dense = nn.Linear(16 * 3 * 3, 9)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = [hidden] * 3
        outputs = []
        x = self.layer1(inputs, hidden[0])
        outputs.append(x)
        x = self.layer2(x, hidden[1])
        outputs.append(x)
        x = self.layer3(x, hidden[2])
        outputs.append(x)

        out = self.dense(x.view(x.size(0), -1))
        return out, outputs
