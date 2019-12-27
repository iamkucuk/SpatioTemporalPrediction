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

model = ConvNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
train_dataset = ConvNetDataset(train_dataset)
validation_dataset = ConvNetDataset(validation_dataset)
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=32),
    "val": DataLoader(validation_dataset, batch_size=32)
}
engine1 = ModelEngine(model, criterion, optimizer)

#%%

engine1.fit(dataloaders)

#%%

model = CNNwithRNN()
optimizer = optim.Adam(model.parameters())
engine2 = ModelEngine(model, criterion, optimizer, isRecurrent=True)
engine2.fit(dataloaders)