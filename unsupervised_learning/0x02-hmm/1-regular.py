#!/usr/bin/env python3
"""
Determine the steady state probailities of a regular
Markov Chain
"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns:
        numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure
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
    s = np.ones((1, n)) / n
    while True:
        s2 = s
        s = np.matmul(s, P)
        if (P <= 0).any():
            return None
        if np.all(s == s2):
            return s
