#!/usr/bin/env python3
"""Neuron that performs binary classification"""
import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""

    def __init__(self, nx):
        "Neuron constructor"
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1+np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = len(self.__A[0])
        cost = -1/m * np.sum(np.multiply(np.log(A), Y) + np.multiply(np.log(1.0000001-A), (1-Y)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = A.round().astype(int)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = len(A[0])
        dZ = A - Y
        dW = np.matmul(X, dZ.T) / m
        db = np.sum(dZ) / m
        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            i += 1
        return self.evaluate(X, Y)
