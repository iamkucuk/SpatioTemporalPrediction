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

        # Gates of GRU as Convolutional layers
        self.reset_gate_current = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.update_gate_current = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.out_gate_current = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)

        self.reset_gate_prev = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate_prev = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate_prev = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, inputs, prev_state):
        prev_state = prev_state if prev_state is not None else self.init_hidden((inputs.size(0),
                                                                                 self.hidden_size,
                                                                                 inputs.size(2),
                                                                                 inputs.size(3)))

        z_l = self.sigmoid(self.update_gate_current(inputs) + self.update_gate_prev(prev_state))
        r_l = self.sigmoid(self.reset_gate_current(inputs) + self.reset_gate_prev(prev_state))
        h_tilda_t = self.tanh(self.out_gate_current(inputs) + self.out_gate_prev(prev_state * r_l))
        out = prev_state * (1 - z_l) + h_tilda_t * z_l

        return out

    def init_hidden(self, size):
        return Variable(torch.zeros(size)).to(self.device)


class CRNN(nn.Module):

    def __init__(self, input_size=1, device="cpu"):
        super(CRNN, self).__init__()
        self.input_size = input_size
        self.layer1 = GRU2DCell(input_size, 32, 3, device)
        self.layer2 = GRU2DCell(32, 64, 3, device)
        # self.layer3 = GRU2DCell(64, 16, 3, device)
        self.dense = nn.Linear(64 * 3 * 3, 9)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = [None, None]
        else:
            hidden[0] = hidden[0][-inputs.size(0):]
            hidden[1] = hidden[1][-inputs.size(0):]
        outputs = []
        x = self.layer1(inputs, hidden[0])
        outputs.append(x)
        x = self.layer2(x, hidden[1])
        outputs.append(x)

        out = self.dense(x.view(x.size(0), -1))
        return out, outputs

    def init_hidden(self, size):
        pass
