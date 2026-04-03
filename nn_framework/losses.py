from .tensor import Tensor
import numpy as np

class Loss:
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

class MSE(Loss):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return (diff ** 2).sum() * (1.0 / y_pred.data.size)

class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        batch_size = y_pred.data.shape[0]
        log_probs = y_pred.log()
        loss = (y_true * log_probs).sum() * (-1.0 / batch_size)
        return loss

class BCE(Loss):
    def forward(self, y_pred, y_true):
        batch_size = y_pred.data.shape[0]
        res = y_true * y_pred.log() + (Tensor(1.0) - y_true) * (Tensor(1.0) - y_pred).log()
        return res.sum() * (-1.0 / batch_size)