import numpy as np
import torch

from CNN_RNN import CNNwithRNN
from CRNN import CRNN
from ConvNet import ConvNet
from utils import cross_validation

dataset = np.load("data/train_set.npy")

# model = ConvNet()
# model = CNNwithRNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CRNN(input_size=10, device=device)
hebe = cross_validation(trainset=dataset, model=model, fold_number=10)

