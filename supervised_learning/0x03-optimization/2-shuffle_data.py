#!/usr/bin/env python3
"""shuffle the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    shuffler = np.random.permutation(len(X))
    X_shuffled = X[shuffler]
    Y_shuffled = Y[shuffler]
    return X_shuffled, Y_shuffled
