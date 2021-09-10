#!/usr/bin/env python3
"""
Update the wheights and biases if a neural network using gradient
descent with L2regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, wheights, cache, alpha, lambtha, L):
    """
    Updates the wheights and biases if a neural network using gradient
    descent with L2regularization
    """
    m = Y.shape[1]
    for i in reversed(range(L)):
        layer = str(i + 1)
        prev_layer = str(i)
        W_key = 'W' + layer
        A_key = 'A' + layer
        A_prev = 'A' + prev_layer
        b_key = 'b' + layer
        if i == L - 1:
            dZ = cache['A' + str(L)] - Y
        else:
            dZ = dA * (1 - (cache[A_key] ** 2))
        dW = (np.matmul(dZ, cache[A_prev].T) / m) + \
            ((lambtha / m) * wheights[W_key])
        db = (np.sum(dZ, axis=1, keepdims=True)) / m
        dA = np.matmul(wheights[W_key].T, dZ)
        wheights[W_key] = wheights[W_key] - alpha * dW
        wheights[b_key] = wheights[b_key] - alpha * db
