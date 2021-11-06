#!/usr/bin/env python3
"""
Perform PCA on a dataset
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performas PCA on a dataset.

    Arguments:
    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that
        the PCA transformation should maintain
    Return:
        the weights matrix, W, that maintains var fraction
            of X‘s original variance
        W is a numpy.ndarray of shape (d, nd) where nd is the
            new dimensionality of the transformed X
    """
    _, S, Vh = np.linalg.svd(X)

    cum = [S[0]]

    for i in range(1, len(S)):
        cum.append(S[i] + cum[-1])

    idx = len(cum) - 1
    for i in range(len(cum)):
        if cum[i] / cum[-1] >= var:
            idx = i
            break

    return Vh.T[:, :i + 1]
