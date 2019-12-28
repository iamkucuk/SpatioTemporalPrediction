#%%

from torch.utils.data import DataLoader
from model_utils import ModelEngine
import numpy as np
import torch
from torch import nn, optim

from CNN_RNN import CNNwithRNN
from CRNN import CRNN
from ConvNet import ConvNet
from utils import train_test_split
from DatasetUtils import *

#%%

dataset = np.load("data/train_set.npy")
trainset, testset = train_test_split(dataset)
train_dataset, validation_dataset = train_test_split(dataset=trainset, train_set_ratio=1 - (len(testset) / len(trainset)))

#%%

model1 = ConvNet()
criterion1 = nn.BCEWithLogitsLoss()
optimizer1 = optim.Adam(model1.parameters())
train_dataset1 = ConvNetDataset(train_dataset)
validation_dataset1 = ConvNetDataset(validation_dataset)
dataloaders1 = {
    "train": DataLoader(train_dataset1, batch_size=16),
    "val": DataLoader(validation_dataset1, batch_size=16)
}
engine1 = ModelEngine(model1, criterion1, optimizer1)

#%%

engine1.fit(dataloaders1)
