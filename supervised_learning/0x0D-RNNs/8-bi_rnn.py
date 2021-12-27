#!/usr/bin/env python3
"""Perform Forward propagation for a bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:

    bi_cell is an instance of BidirectinalCell that will be
        used for the forward propagation
    X is the data to be used, given as a numpy.ndarray of
            shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction,
            given as a numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction,
        given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    tmax, m, _ = X.shape
    _, h = h_0.shape
    Hf = np.zeros((tmax + 1, m, h))
    Hb = np.zeros((tmax + 1, m, h))
    Hf[0] = h_0
    Hb[tmax] = h_t

    for t in range(tmax):
        Hf[t + 1] = bi_cell.forward(Hf[t], X[t])

    for t in range(tmax - 1, -1, -1):
        Hb[t] = bi_cell.backward(Hb[t + 1], X[t])

    H = np.concatenate((Hf[1:], Hb[0:tmax]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
