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
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

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
