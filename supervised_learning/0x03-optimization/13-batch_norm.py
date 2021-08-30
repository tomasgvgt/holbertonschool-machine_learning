#!/usr/bin/env python3
"""
Normalize an unactivated output of a neural network
using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a
    neural network using batch normalization
    """
    Z_mean = Z.mean(0)
    Z_var = Z.var(0)
    dividend = Z - Z_mean
    divisor = np.sqrt(Z_var) + epsilon
    Z_normalized = dividend / divisor
    Z_batch_normalized = gamma * Z_normalized + beta
    return Z_batch_normalized
