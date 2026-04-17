import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Initialize the KNN model.
 
        Arguments:
            k (int): number of nearest neighbors to use
            task_kind (str): "classification" or "regression"
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels, visualize=False):
        """
        Trains the model, returns predicted labels for training data.

        KNN has no real training step: we simply memorize the training set.
        We then predict on the training data itself and return those predictions.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
            visualize(bool): unused, kept for API consistency with other methods
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels
        return self._predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        assert hasattr(self, 'training_data'), \
            "Model not trained yet! Call fit() before predict()."
        return self._predict(test_data)
    
    def _predict(self, data):
        """
        Core prediction logic shared by fit() and predict().
 
        Computes Euclidean distances from each query point to all training
        points, selects the k nearest, then aggregates by majority vote
        (classification) or mean (regression).
 
        Arguments:
            data (np.array): query points of shape (M, D)
        Returns:
            pred_labels (np.array): predicted labels of shape (M,)
        """
        # data:(M, D)
        # training_data:(N, D)
        data_sq = np.sum(data ** 2, axis=1, keepdims=True)            # (M, 1)
        train_sq = np.sum(self.training_data ** 2, axis=1, keepdims=True)  # (N, 1)
        cross = data @ self.training_data.T                            # (M, N)
        sq_dists = data_sq + train_sq.T - 2.0 * cross                 # (M, N)
 
        sq_dists = np.maximum(sq_dists, 0.0)
        distances = np.sqrt(sq_dists)                                  
        k_nearest_idx = np.argsort(distances, axis=1)[:, :self.k] 
        k_labels = self.training_labels[k_nearest_idx]                
 
        if self.task_kind == "classification":
            # return the most frequent label among the k neighbors
            pred_labels = np.array([
                np.bincount(row.astype(int)).argmax()
                for row in k_labels
            ])
        else:
            # return the mean of the k neighbors' values
            pred_labels = k_labels.mean(axis=1)
 
        return pred_labels