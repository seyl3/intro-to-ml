import numpy as np


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        """
        The Linear Regression does not require any hyperparameter, 
        since this is without regularization. Optimal weights 
        are computed directly with the closed form solution. 
        """

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Adds a bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        X = np.hstack([np.ones((training_data.shape[0], 1)), training_data])
        self.W = np.linalg.pinv(X) @ training_labels # or equivalently np.linalg.lstsq(X, training_labels, rcond=None)[0]

        pred_labels = X @ self.W
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        assert hasattr(self, 'W'), "The model not trained yet! You need to call fit() first"
        X = np.hstack([np.ones((test_data.shape[0], 1)), test_data])

        pred_labels = X @ self.W
        return pred_labels
