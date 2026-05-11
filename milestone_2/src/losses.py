import numpy as np

from milestone_1.src.utils import mse_fn


class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        N = y_pred.size
        return mse_fn(y_pred, y_true)

    @staticmethod
    def gradient(y_true, y_pred):
        N = y_pred.size
        return (2 / N) * (y_pred - y_true)
