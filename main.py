import numpy as np
from utils import sequential_fold_generator

dataset = np.load("data/train_set.npy")

for x, y in sequential_fold_generator(dataset=dataset, fold_number=10):
    print(x.shape)
    print(y.shape)

