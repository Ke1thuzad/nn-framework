from .module import Module
from .tensor import Tensor
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2 / in_features)
        self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)))
        self.b = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
