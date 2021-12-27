#!/usr/bin/env python3
"""Class BidirectionalCell that represents a bidirectional cell of an RNN"""
import numpy as np


class BidirectionalCell:
    """Class that represents a bidirectional cell of an RNN"""
    def __init__(self, i, h, o):
        """
        class constructor def __init__(self, i, h, o):
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
                that represent the weights and biases of the cell
            Whf and bhf are for the hidden states in the forward direction
            Whb and bhb are for the hidden states in the backward direction
            Wy and by are for the outputs
        The weights should be initialized using a random normal distribution
            in the order listed above
        The weights will be used on the right side for matrix multiplication
            The biases should be initialized as zeros
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the forward direction for one time step
            x_t is a numpy.ndarray of shape (m, i) that contains
                the data input for the cell
            m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h)
                containing the previous hidden state
            Returns: h_next, the next hidden state
        """
        mat1 = np.concatenate((h_prev, x_t), axis=-1)
        Whf, bhf = self.Whf, self.bhf
        h_next = np.tanh(mat1 @ Whf + bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction for one time step
            x_t is a numpy.ndarray of shape (m, i) that contains
                    the data input for the cell
                m is the batch size for the data
            h_next is a numpy.ndarray of shape (m, h) containing
                the next hidden state
        Returns: h_pev, the previous hidden state
        """
        mat1 = np.concatenate((h_next, x_t), axis=-1)
        Whb, bhb = self.Whb, self.bhb
        h_pev = np.tanh(mat1 @ Whb + bhb)

        return h_pev

    def output(self, H):
        """
        calculates all outputs for the RNN
        H is a numpy.ndarray of shape (t, m, 2 * h) that contains
            the concatenated hidden states from both directions,
            excluding their initialized states
        t is the number of time steps
        m is the batch size for the data
        h is the dimensionality of the hidden states
        """
        Wy, by = self.Wy, self.by
        Y = H @ Wy + by[np.newaxis, :, :]
        for i in range(len(Y)):
            Y[i] = self.softmax(Y[i])
        return Y

    @staticmethod
    def softmax(x):
        """Softmax function"""
        x_max = np.max(x, axis=1, keepdims=True)
        x_e = np.exp(x - x_max)
        return x_e / np.sum(x_e, axis=1, keepdims=True)
