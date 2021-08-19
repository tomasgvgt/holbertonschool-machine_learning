#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """deep neural network performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.L):
            W_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.weights[W_key] = np.random.randn(
                    layers[i], nx) * np.sqrt(2/nx)
            else:
                self.weights[W_key] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2/layers[i - 1])
            self.weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the deep neural network"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            A_prev = "A" + str(i)
            W_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            A_current = "A" + str(i + 1)
            Z = np.matmul(self.__weights[W_key],
                          self.__cache[A_prev]) + self.__weights[b_key]
            if i == self.__L - 1:
                t = np.exp(Z)
                self.__cache[A_current] = t / np.sum(t, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    self.__cache[A_current] = 1 / (1 + np.exp(-Z))
                elif self.__activation == 'tanh':
                    self.__cache[A_current] = np.tanh(Z)
        return self.__cache[A_current], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = len(A[0])
        cost = -1/m * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions"""
        A, cache = self.forward_prop(X)
        A_max = np.amax(A, axis=0)
        cost = self.cost(Y, A)
        Y_hat = np.where(A == A_max, 1, 0)
        return Y_hat, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = len(Y[0])
        for i in reversed(range(self.__L)):
            A_current = "A" + str(i + 1)
            A_prev = "A" + str(i)
            W_current = "W" + str(i + 1)
            b_current = "b" + str(i + 1)
            if i == self.__L - 1:
                dZ = cache[A_current] - Y
            else:
                if self.__activation == 'sig':
                    dZ = dA * ((cache[A_current]) * (1 - cache[A_current]))
                elif self.__activation == 'tanh':
                    dZ = dA * (1 - (cache[A_current] ** 2))
            dW = np.matmul(dZ, cache[A_prev].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.matmul(self.__weights[W_current].T, dZ)
            self.__weights[W_current] = self.__weights[W_current] - alpha * dW
            self.__weights[b_current] = self.__weights[b_current] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        iteration_list = []
        cost_list = []
        for i in range(iterations + 1):
            Y_hat, cost = self.evaluate(X, Y)
            self.gradient_descent(Y, self.__cache, alpha)
            if i % step == 0:
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                iteration_list.append(i)
                cost_list.append(cost)

        if graph is True:
            plt.plot(iteration_list, cost_list, 'blue')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return Y_hat, cost

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
