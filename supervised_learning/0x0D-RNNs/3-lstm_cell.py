#!/usr/bin/env python3
"""
Class LSTMcell that represents a gated recurrent unit
"""
import numpy as np


class LSTMCell:
    """
    Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
                that represent the weights and biases of the cell
            Wzand bz are for the update gate
            Wrand br are for the reset gate
            Whand bh are for the intermediate hidden state
            Wyand by are for the output
        The weights should be initialized using a random normal distribution
                in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """Computes softmax values for each set of scores in x"""
        x_max = np.max(x, axis=1, keepdims=True)
        x_e = np.exp(x - x_max)
        return x_e / np.sum(x_e, axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
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
        Wf, Wu, Wc, Wo, Wy = self.Wf, self.Wu, self.Wc, self.Wo, self.Wy
        bf, bu, bc, bo, by = self.bf, self.bu, self.bc, self.bo, self.by
        mat1 = np.concatenate((h_prev, x_t), axis=-1)
        f_t = self.sigmoid(mat1 @ Wf + bf)
        u_t = self.sigmoid(mat1 @ Wu + bu)
        o_t = self.sigmoid(mat1 @ Wo + bo)
        mat2 = np.tanh(mat1 @ Wc + bc)
        Ct = f_t * c_prev + u_t * mat2
        ht = o_t * np.tanh(Ct)
        y = self.softmax(ht @ Wy + by)

        return ht, Ct, y
