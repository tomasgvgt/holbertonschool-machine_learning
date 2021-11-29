#!/usr/bin/env python3
"""
Determine if a markov chain is absorving
"""
import numpy as np


def absorbing(P):
    """
    Determine if a markov chain is absorbing:

    P is a is a square 2D numpy.ndarray of shape (n, n)
            representing the standard transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns:
        True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if P.ndim != 2:
        return None
    if (P < 0).any() or (P > 1).any():
        return None

    n = P.shape[0]
    d = np.where(np.diag(P) == 1)[0]
    if (d == 1).any():
        return True
    if not (d == 1).all():
        return False
    for i in range(n):
        if P[i][i] == 1:
            return True
    return False
