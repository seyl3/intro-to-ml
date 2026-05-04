import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        ### WRITE YOUR CODE HERE
        
        self.dimensions = dimensions
        self.activations = activations
        
        self.weights = []
        self.biases = []
        
        # pour créer les connexions
        for i in range(len(dimensions) - 1):
            n_in = dimensions[i]
            n_out = dimensions[i+1]
            
            # initialisation aléatoire des poids
            w = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
            self.weights.append(w)
            
            # un biais par neurone de sortie
            b = np.zeros((1, n_out))
            self.biases.append(b)

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        ### WRITE YOUR CODE HERE
        
         # couche 0 -> données d'entrée
        a = {0: x}
        z = {}

        # parcourt de chaque transition entre les couches
        for i in range(len(self.weights)):
            # pré-activation
            z[i+1] = a[i] @ self.weights[i] + self.biases[i]
            
            # activation
            a[i+1] = self.activations[i].forward(z[i+1])

        return z, a


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        ### WRITE YOUR CODE HERE
        
        _, activations = self.feed_forward(x)
        
        # renvoie de la dernière couche uniquement
        return activations[len(self.weights)]


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        ### WRITE YOUR CODE HERE
        
        dw = {}
        db = {}
        deltas = {}
        L = len(self.weights)
        
        # erreur à la sortie
        output_pred = a[L]
        deltas[L] = loss.gradient(y_true, output_pred) * self.activations[L-1].gradient(z[L])

        # rétropropagation de l'erreur dans les couches cachées
        for i in range(L - 1, 0, -1):
            deltas[i] = (deltas[i+1] @ self.weights[i].T) * self.activations[i-1].gradient(z[i])

        # Calcul des gradients dw et db pondérés par la taille du batch
        for i in range(L):
            dw[i] = (a[i].T @ deltas[i+1]) / a[i].shape[0]
            db[i] = np.mean(deltas[i+1], axis=0, keepdims=True)

        return dw, db


    def update_w_b(self, index, dw, db, learning_rate):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        ### WRITE YOUR CODE HERE
        
        self.weights[index] -= learning_rate * dw
        self.biases[index] -= learning_rate * db
        
    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        ### WRITE YOUR CODE HERE
        
        n_samples = x.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y_true[indices]
            
            # parcours par mini batch
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # calcul des activations
                z, a = self.feed_forward(x_batch)
                
                # calcul des gradients
                dw, db = self.back_prop(z, a, y_batch, loss)
                
                # Mise à jour de chaque couche
                for layer_idx in range(len(self.weights)):
                    self.update_w_b(layer_idx, dw[layer_idx], db[layer_idx], learning_rate)
