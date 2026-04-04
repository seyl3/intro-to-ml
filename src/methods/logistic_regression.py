import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


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
        C = training_labels.shape[1]
        self.W = np.random.normal(0, 0.1, (D, C))
        for it in range(self.max_iters):
            gradient = self._gradient(training_data, training_labels)
            weights = weights - self.lr * gradient

            pred_labels = self.predict(training_data)
            if self._acc(pred_labels, np.argmax(training_labels, axis=1)) == 100:
                break
            #logging and plotting
        #    if print_period and it % print_period == 0:
         #       print('loss at iteration', it, ":", loss_logistic_multi(training_data, training_labels, weights))
         #   if plot_period and it % plot_period == 0:
         #       fig = helpers.visualize_predictions(data=training_data, labels_gt=helpers.onehot_to_label(ltraining_abels), labels_pred=predictions, title="iteration "+ str(it))
                
        #fig = helpers.visualize_predictions(data=training_data, labels_gt=helpers.onehot_to_label(training_labels), labels_pred=predictions, title="final model")


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

        pred_labels = np.argmax(self._softmax(test_data, self.W), axis=1)

        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return pred_labels

    def _softmax(self, data):
        e = np.exp(data @ self.W)
        return e / np.sum(e, axis=1)[:, np.newaxis]

    def _loss(self, data, labels):
        return -np.sum(labels * np.log(self._softmax(data)))

    def _gradient(self, data, labels):
        return data.T @ (self._softmax(data, self.W) - labels)
    
    def _acc(self, labels_pred, labels_gt):
        diff = np.count_nonzero(labels_gt - labels_pred)
        correct = np.shape(labels_pred)[0] - diff
        return correct / np.shape(labels_gt)[0] * 100
