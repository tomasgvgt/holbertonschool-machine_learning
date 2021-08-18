#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.L):
            W_key = "W" + str(i + 1)
            b_key = "b" + str(i + 1)
            if not isinstance(layers[i], int):
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
            self.__cache[A_current] = 1 / (1 + np.exp(-Z))
        return self.__cache[A_current], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = len(A[0])
        cost = -1/m * np.sum(np.multiply(np.log(A), Y) + np.multiply(
            np.log(1.0000001-A), (1-Y)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        Y_hat = A.round().astype(int)
        return Y_hat, cost
