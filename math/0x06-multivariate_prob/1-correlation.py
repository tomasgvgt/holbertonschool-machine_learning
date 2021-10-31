#!/usr/bin/env python3
"""
Calculate a correlation matrix
"""
import numpy as np


def correlation(C):
    """
    Calculates the correlation of a matrix

    Attributes:
    C is a numpy.ndarray of shape (d, d) containing a covariance matrix
        d is the number of dimensions
    Requisites:
        If C is not a numpy.ndarray,
            raise a TypeError with the message C must be a numpy.ndarray
        If C does not have shape (d, d),
            raise a ValueError with the message C must be a 2D square matrix
    Return:
    a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) != np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    variance = np.sqrt(np.diag(1 / np.diag(C)))
    correlation = np.matmul(np.matmul(variance, C), variance)
    return correlation
