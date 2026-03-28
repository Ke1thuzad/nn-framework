import numpy as np
import pandas as pd
from .tensor import Tensor

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def map(self, func):
        self.x = func(self.x)
        return self

    @classmethod
    def from_csv(cls, path, target_cols):
        df = pd.read_csv(path)
        y = df[target_cols].values
        x = df.drop(columns=target_cols).values
        return cls(x, y)

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            x_batch = self.dataset.x[batch_indices]
            y_batch = self.dataset.y[batch_indices]

            yield Tensor(x_batch), Tensor(y_batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size