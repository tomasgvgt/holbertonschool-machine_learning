#!/usr/bin/env python3
"""
Create a cell of a simple RNN
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wh, Wy, bh, by
                that represent the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
        The weights should be initialized using a random normal distribution
            in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """Computes softmax values for each set of scores in x"""
        x_max = np.max(x, axis=1, keepdims=True)
        x_e = np.exp(x - x_max)
        return x_e / np.sum(x_e, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Perfomrs forward propagation for one time step.

        x_t is a numpy.ndarray of shape (m, i) that
            contains the data input for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h)
            containing the previous hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        h_x = np.concatenate((h_prev, x_t), axis=-1)
        Wh, Wy, bh, by = self.Wh, self.Wy, self.bh, self.by
        h_next = np.tanh(np.matmul(h_x, Wh) + bh)
        y = self.softmax(np.matmul(h_next, Wy) + by)
        return h_next, y
