#!/usr/bin/env python3
"""
Updates the weights of a neural network with
Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with
    Dropout regularization using gradient descent
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        layer = str(i + 1)
        prev_layer = str(i)
        A_key = 'A' + layer
        d_key = 'D' + layer
        W_key = 'W' + layer
        b_key = 'b' + layer
        if i == (L - 1):
            dZ = cache['A' + str(L)] - Y
        else:
            dZ = dA * 1 - (cache[A_key] ** 2) * cache[d_key]
            dZ /= keep_prob
        dW = np.matmul(dZ, cache['A' + prev_layer].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.matmul(weights[W_key].T, dZ)
        weights[W_key] = weights[W_key] - alpha * dW
        weights[b_key] = weights[b_key] - alpha * db
