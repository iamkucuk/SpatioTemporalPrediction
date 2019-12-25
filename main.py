import numpy as np

from ConvNet import ConvNet
from utils import cross_validation

dataset = np.load("data/train_set.npy")

model = ConvNet()

hebe = cross_validation(trainset=dataset, model=model, fold_number=10)

