#!/usr/bin/env python3
"""calculate the normalization (standardization) constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    standard_deviation = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    return mean, standard_deviation
