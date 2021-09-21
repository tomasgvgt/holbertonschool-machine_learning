#!/usr/bin/env python3
"""
Perform forward propagation over a
convolutional layer of a NN
"""
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """
    Performs forward prop over a convolutional layer of a NN.
    Attributes:
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer

        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer

        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output

        b is a numpy.ndarray of shape (1, 1, 1, c_new)
        containing the biases applied to the convolution

        activation: activation function applied to the convolution.
        padding: string that is either same or valid,
            indicating the type of padding used.

        stride: tuple of (sh, sw) containing the strides
        for the convolution
            sh: stride for the height
            sw: stride for the width
    Returns:
        Output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    if padding == 'valid':
        ph = 0
        pw = 0

    h_out = (h_prev - kh + 2 * ph) // sh + 1
    w_out = (w_prev - kw + 2 * pw) // sw + 1

    image = np.pad(
                   A_prev,
                   pad_width=((0, 0),
                              (ph, ph),
                              (pw, pw),
                              (0, 0)),
                   mode='constant',
                   constant_values=0)
    output = np.zeros((m, h_out, w_out, c_new))

    for y in range(h_out):
        for x in range(w_out):
            for cn in range(c_new):
                output[:, y, x, cn] = \
                    (W[:, :, :, cn] *
                     image[:,
                     y * sh: y * sh + kh,
                     x * sw: x * sw + kw,
                     :]).sum(axis=(1, 2, 3))
                output[:, y, x, cn] = \
                    (activation(output[:, y, x, cn] +
                                b[0, 0, 0, cn]))
    return output