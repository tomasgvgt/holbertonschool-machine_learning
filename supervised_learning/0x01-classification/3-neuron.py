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
        cost = -1/m * np.sum(np.multiply(np.log(A), Y) + np.multiply(
            np.log(1.0000001 - A), (1 - Y)))
        return cost
