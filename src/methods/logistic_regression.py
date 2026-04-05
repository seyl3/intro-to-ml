import numpy as np

from ..utils import get_n_classes, label_to_onehot


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.W = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        D = training_data.shape[1]
        C = get_n_classes(training_labels)
        training_labels_onehot = label_to_onehot(training_labels, C)
        self.W = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = self._gradient(training_data, training_labels_onehot)
            self.W = self.W - self.lr * gradient

            pred_labels = self.predict(training_data)
            if self._acc(pred_labels, training_labels) == 100:
                break
            
        if self.max_iters == 0:
            pred_labels = self.predict(training_data)

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        if (self.W is None): raise ValueError("Model has not been trained yet")

        pred_labels = np.argmax(self._softmax(test_data), axis=1)

        return pred_labels

    def _softmax(self, data):
        scores = data @ self.W
        scores = scores - np.max(scores, axis=1, keepdims=True)
        e = np.exp(scores)
        return e / np.sum(e, axis=1, keepdims=True)

    def _loss(self, data, labels):
        probs = self._softmax(data)
        return -np.mean(np.sum(labels * np.log(probs + 1e-12), axis=1))

    def _gradient(self, data, labels):
        N = data.shape[0]
        return data.T @ (self._softmax(data) - labels) / N
    
    def _acc(self, labels_pred, labels_gt):
        return np.mean(labels_pred == labels_gt) * 100
