import time
import numpy as np

from .tensor import Tensor
from .module import Module

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Model:
    def __init__(self, network, loss_fn, optimizer, metrics=None):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics if metrics else []

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        output = self.forward(x)
        return output.data

    def _train_step(self, x, y):
        self.optimizer.zero_grad()

        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)

        loss.backward()
        self.optimizer.step()

        return loss.data

    def fit(self, dataloader, epochs=10, val_loader=None):
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0

            for x_batch, y_batch in dataloader:
                loss_val = self._train_step(x_batch, y_batch)
                train_loss += loss_val

            avg_loss = train_loss / len(dataloader)

            log_str = f"Epoch [{epoch + 1}/{epochs}] - loss: {avg_loss:.4f}"

            eval_loader = val_loader if val_loader else dataloader
            eval_results = self.evaluate(eval_loader)

            for metric_name, value in eval_results.items():
                log_str += f" - {metric_name}: {value:.4f}"

            duration = time.time() - start_time
            print(f"{log_str} - {duration:.2f}s")

    def evaluate(self, dataloader):
        results = {}
        all_preds = []
        all_trues = []

        for x, y in dataloader:
            preds = self.predict(x)
            all_preds.append(preds)
            all_trues.append(y.data)

        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)

        for metric in self.metrics:
            name = metric.__class__.__name__.upper()
            results[name] = metric.calculate(all_preds, all_trues)

        return results
