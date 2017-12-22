import numpy as np
from utils import sigmoid, sigmoid_backward, relu, relu_backward
import pickle
import matplotlib.pyplot as plt


class NN:
    """
    Deep Neural Network having L layers.
    dims_layer : list containing dimensions of layers, size = L+1
    """

    def __init__(self, dims_layer):
        self.parameters = None
        self.dims_layer = dims_layer

    def initialize_parameters(self):
        self.parameters = {}
        L = len(self.dims_layer)

        # Initialize weights and bias for all layers
        for l in range(1, L):
            # initialize weight to random small value
            # dimensions of weight are W[l] = (dims_layer[l], dims_layer[l-1])
            self.parameters['W' + str(l)] = np.random.randn(self.dims_layer[l],
                                                            self.dims_layer[l - 1]) * 0.01
            # initialize bias to zero values
            # dimensions of bias vector is b[l] = (self.dims_layer[l], 1)
            self.parameters['b' + str(l)] = np.zeros((self.dims_layer[l], 1))

    def _linear_forward(self, A, W, b):
        """
        Forward propagation.
        """
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def _linear_backward(self, dZ, cache):
        """
        Backward propagation.
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db

    def _linear_activation_forward(self, A_prev, W, b, activation):
        """
        Get Activation A for a layer with given parameters.
        Returns cache to use in backward propagation.
        """
        if activation == "sigmoid":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def _linear_activation_backward(self, dA, cache, activation):
        """
        Get Activation for backward propagation.
        Returns gradients.
        """
        linear_cache, activation_cache = cache
        if activation == 'relu':
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def forward_propagation(self, X):
        """
        Forward propagate on this neuron for given input sets X.
        Returns final output AL of output layer.
        """
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self._linear_activation_forward(
                A_prev, self.parameters["W" + str(l)], self.parameters["b" + str(l)], activation='relu')
            caches.append(cache)

        AL, cache = self._linear_activation_forward(
            A, self.parameters["W" + str(L)], self.parameters["b" + str(L)], activation='sigmoid')
        caches.append(cache)

        return AL, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]

        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(
            L)] = self._linear_activation_backward(dAL, current_cache, activation='sigmoid')
        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(
                grads['dA' + str(l + 2)], current_cache, activation='relu')
            grads['dA' + str(l + 1)] = dA_prev_temp
            grads['dW' + str(l + 1)] = dW_temp
            grads['db' + str(l + 1)] = db_temp
        return grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters["W" + str(l + 1)] -= learning_rate * \
                grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= learning_rate * \
                grads["db" + str(l + 1)]

    def compute_cost(self, AL, Y):
        """
        Compute cost for given output AL, Y is the actual output.
        """
        m = Y.shape[1]
        cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y),
                                                                np.log(1 - AL)), axis=1, keepdims=True) / m
        return np.squeeze(cost)

    def fit(self, X, Y, num_iterations=20000, print_cost=False, learning_rate=0.0085):
        """
        fit the dataset to the model.
        """
        costs = []
        self.learning_rate = learning_rate
        for i in range(0, num_iterations):
            AL, caches = self.forward_propagation(X)

            cost = self.compute_cost(AL, Y)

            grads = self.backward_propagation(AL, Y, caches)

            self.update_parameters(grads, self.learning_rate)

            if i % 100 == 0 and print_cost:
                print('\rCost after iteration {}: {}'.format(i, np.squeeze(cost)))
                costs.append(cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration (per tens)')
        plt.title('Learning rate = ' + str(learning_rate))
        plt.show()

    def dump(self, filename):
        with open(filename, 'wb+') as fo:
            pickle.dump([self.dims_layer, self.parameters], fo)

    def load(self, filename):
        with open(filename, 'rb') as fo:
            self.dims_layer, self.parameters = pickle.load(fo)


if __name__ == '__main__':
    dims = [32 * 32 * 3, 37, 41, 37, 23, 19, 10]
    deep_model = NN(dims)
    deep_model.initialize_parameters()
    # datasets = ['datasets/data_batch_' + str(i) for i in range(1, 2)]
    dataset_file = 'datasets/data_batch_1'
    dict_data = None
    with open(dataset_file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    X = dict_data[b'data'].T
    Y_in = dict_data[b'labels']
    Y = np.zeros((10, len(Y_in)))
    for i in range(0, Y.shape[1]):
        Y[Y_in[i]][i] = 1
    test_dataset = 'datasets/test_batch'
    with open(test_dataset, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    X_test = dict_data[b'data'].T
    Y_test = np.array(dict_data[b'labels'])
    Y_test = Y_test.reshape((Y_test.shape[0], 1))
    dims = [32 * 32 * 3, 37, 41, 37, 23, 19, 10]
    deep_model.fit(X, Y, num_iterations=30000, print_cost=True)
    deep_model.dump('deep_model.pkl')
