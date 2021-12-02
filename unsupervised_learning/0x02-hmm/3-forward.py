#!/usr/bin/env python3
"""
Perform the forward algorithm for a hidden Markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    performs the forward algorithm for a hidden markov model:

    Observation is a numpy.ndarray of shape (T,) that contains the index
        of the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
            probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the
            hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
            transition probabilities
        Transition[i, j] is the probability of transitioning from
            the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state
    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward
                path probabilities
            F[i, j] is the probability of being in hidden state i at
                time j given the previous observations
    """
    if type(Observation) is not np.ndarray:
        return None, None
    if Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray:
        return None, None
    if Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray:
        return None, None
    if Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Transition.shape[0] != Emission.shape[0]:
        return None, None
    if type(Initial) is not np.ndarray:
        return None, None
    if Initial.ndim != 2:
        return None, None
    if Initial.shape != (Emission.shape[0], 1):
        return None, None
    N = Emission.shape[0]
    T = Observation.shape[0]
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for i in range(1, T):
        s = np.matmul(F[:, i - 1].T, Transition)
        F[:, i] = s * Emission[:, Observation[i]]
    P = np.sum(F[:, -1])
    return P, F
