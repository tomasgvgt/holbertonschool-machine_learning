#!/usr/bin/env python3
"""
Class GRUCell that represents a gated recurrent unit
"""
import numpy as np


class GRUCell:
    """
    REpresents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh,
                by that represent the weights and biases of the cell
            Wzand bz are for the update gate
            Wrand br are for the reset gate
            Whand bh are for the intermediate hidden state
            Wyand by are for the output
        The weights should be initialized using a random normal distribution
        `   in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """Softmax function"""
        x_max = np.max(x, axis=1, keepdims=True)
        x_e = np.exp(x - x_max)
        return x_e / np.sum(x_e, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """Sigmoid function"""
        return (1 / (1 + np.exp(-x)))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step
        x_t is a numpy.ndarray of shape (m, i) that contains the data
                input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the
            previous hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        Wz, Wr, Wh, Wy = self.Wz, self.Wr, self.Wh, self.Wy
        bz, br, bh, by = self.bz, self.br, self.bh, self.by
        mat1 = np.concatenate((h_prev, x_t), axis=-1)
        z_t = self.sigmoid(mat1 @ Wz + bz)
        r_t = self.sigmoid(mat1 @ Wr + br)
        mat2 = np.concatenate((r_t * h_prev, x_t), axis=-1)
        ht = np.tanh(mat2 @ Wh + bh)
        ht = (1 - z_t) * h_prev + z_t * ht
        y = self.softmax(ht @ Wy + by)

        return ht, y
