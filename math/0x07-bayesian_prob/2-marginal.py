#!/usr/bin/env python3
"""
Calculates the marginal probability of obtaining the data
"""
import numpy as np


def marginal(x, n, P, Pr):
    """
    alculates the marginal probability of obtaining the data:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of patients developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs about P
    If n is not a positive integer, raise a ValueError with
        the message n must be a positive integer
    If x is not an integer that is greater than or equal
        to 0, raise a ValueError
        with the message x must be an integer that is
        greater than or equal to 0
    If x is greater than n, raise a ValueError with the
        message x cannot be greater than n
    If P is not a 1D numpy.ndarray, raise a TypeError with
        the message P must be a 1D numpy.ndarray
    If Pr is not a numpy.ndarray with the same shape as P, raise a
        TypeError with the message Pr must be a numpy.ndarray
        with the same shape as P
    If any value in P or Pr is not in the range [0, 1], raise a ValueError
        with the message All values in {P} must be in the
        range [0, 1] where {P} is the incorrect variable
    If Pr does not sum to 1, raise a ValueError with the
        message Pr must sum to 1
    All exceptions should be raised in the above order
    Returns:
        the marginal probability of obtaining x and n
    """
    if type(n) != int or n < 0:
        raise ValueError('n must be a positive integer')
    if type(x) != int or x < 0:
        e = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(e)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) != np.ndarray or len(P.shape) != len(Pr.shape):
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError('All values in P must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    n_factorial = np.math.factorial(n)
    x_factorial = np.math.factorial(x)
    n_minus_x_factorial = np.math.factorial(n - x)
    fact = n_factorial / (x_factorial * n_minus_x_factorial)
    likelihood = fact * np.power(P, x) * np.power((1 - P), (n - x))
    intersection = likelihood * Pr
    return np.sum(intersection)
