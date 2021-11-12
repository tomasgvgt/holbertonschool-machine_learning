#!/usr/bin/env python3
"""
You are conducting a study on a revolutionary cancer drug and are looking
to find the probability that a patient who takes this drug will develop
severe side effects. During your trials, n patients take the drug and
x patients develop severe side effects. You can assume that x follows
a binomial distribution.

Write a function def likelihood(x, n, P):
that calculates the likelihood of obtaining this data given various
hypothetical probabilities of developing severe side effects
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining this data given various
    hipothetical probabilities of developing severe side effects.

    Arguments:
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects
    If n is not a positive integer, raise a ValueError
        with the message n must be a positive integer
    If x is not an integer that is greater than or equal to 0,
        raise a ValueError with the message x must be an integer
        that is greater than or equal to 0
    If x is greater than n, raise a ValueError with the
        message x cannot be greater than n
    If P is not a 1D numpy.ndarray, raise a TypeError with the
        message P must be a 1D numpy.ndarray
    If any value in P is not in the range [0, 1], raise a
        ValueError with the message All values in P must be
        in the range [0, 1]
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) != int or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError('All values in P must be in the range [0, 1]')

    n_factorial = np.math.factorial(n)
    x_factorial = np.math.factorial(x)
    n_minus_x_factorial = np.math.factorial(n - x)
    fact = n_factorial / (x_factorial * n_minus_x_factorial)
    likelihood = fact * np.power(P, x) * np.power((1 - P), (n - x))
    return likelihood
