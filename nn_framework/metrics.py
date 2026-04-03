import numpy as np

class Metric:
    def __call__(self, y_pred, y_true):
        return self.calculate(y_pred, y_true)


class Accuracy(Metric):
    def calculate(self, y_pred, y_true):
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            p = np.argmax(y_pred, axis=1)
            t = np.argmax(y_true, axis=1)
        else:
            p = (y_pred > 0.5).astype(int)
            t = y_true.astype(int)
        return np.mean(p == t)


class Precision(Metric):
    def calculate(self, y_pred, y_true):
        p = (y_pred > 0.5).astype(int)
        tp = np.sum((p == 1) & (y_true == 1))
        fp = np.sum((p == 1) & (y_true == 0))
        return tp / (tp + fp + 1e-15)


class Recall(Metric):
    def calculate(self, y_pred, y_true):
        p = (y_pred > 0.5).astype(int)
        tp = np.sum((p == 1) & (y_true == 1))
        fn = np.sum((p == 0) & (y_true == 1))
        return tp / (tp + fn + 1e-15)


class F1Score(Metric):
    def __init__(self):
        self.precision = Precision()
        self.recall = Recall()

    def calculate(self, y_pred, y_true):
        prec = self.precision.calculate(y_pred, y_true)
        rec = self.recall.calculate(y_pred, y_true)
        return 2 * (prec * rec) / (prec + rec + 1e-15)


class MAE(Metric):
    def calculate(self, y_pred, y_true):
        return np.mean(np.abs(y_pred - y_true))


class R2Score(Metric):
    def calculate(self, y_pred, y_true):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-15))