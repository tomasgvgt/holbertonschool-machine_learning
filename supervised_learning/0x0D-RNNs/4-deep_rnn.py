#!/usr/bin/env python3
"""
Perform Forward Propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l
        that will be used for the forward propagation
        l is the number of layers
    X is the data to be used, given as a numpy.ndarray
            of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray
            of shape (l, m, h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    tmax, m, _ = X.shape
    l, _, h = h_0.shape

    H = np.zeros((tmax + 1, l, m, h))
    H[0] = h_0

    for t in range(tmax):
        for layer in range(l):
            if layer == 0:
                hnext, y = rnn_cells[layer].forward(H[t, layer], X[t])
            else:
                hnext, y = rnn_cells[layer].forward(H[t, layer], hnext)

            H[t + 1, layer, ...] = hnext

            if layer == l - 1:
                if t == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

    Y = Y.reshape(tmax, m, Y.shape[-1])

    return H, Y
