from .module import Module
from .tensor import Tensor
import numpy as np

class Identity(Module):
    def forward(self, x):
        return x

class ReLU(Module):
    def forward(self, x):
        out_data = np.maximum(0, x.data)
        out = Tensor(out_data, (x,), 'ReLU')
        def _backward(): x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out

class LeakyReLU(Module):
    def __init__(self, alpha=0.01): self.alpha = alpha
    def forward(self, x):
        out_data = np.where(x.data > 0, x.data, x.data * self.alpha)
        out = Tensor(out_data, (x,), 'LeakyReLU')
        def _backward():
            dx = np.where(x.data > 0, 1, self.alpha)
            x.grad += dx * out.grad
        out._backward = _backward
        return out

class Sigmoid(Module):
    def forward(self, x):
        return Tensor(1.0) / (Tensor(1.0) + (x * -1.0).exp())

class Tanh(Module):
    def forward(self, x):
        return x.tanh()

class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        shifted_x = x - x.max(axis=self.axis, keepdims=True)
        exps = shifted_x.exp()
        return exps / exps.sum(axis=self.axis, keepdims=True)