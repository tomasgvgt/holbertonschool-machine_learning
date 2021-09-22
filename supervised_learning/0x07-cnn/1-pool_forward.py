#!/usr/bin/env python3
"""
Perform forward propagation over a pooling layer of a newral nerwork
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Arguments:

    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
        m: number of examples
        h_prev: height of the previous layer
        w_prev: width of the previous layer
        c_prev: number of channels in the previous layer
    kernel_shape: tuple of (kh, kw) containing the size
            of the kernel for the pooling
        kh: kernel height
        kw: kernel width
    stride: tuple of (sh, sw) containing the strides for the pooling
        sh: stride for the height
        sw: stride for the width
    mode: string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively

    Returns:
        The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, c_prev))

    for h in range(h_out):
        for w in range(w_out):
            x1 = h * sw
            x2 = h * sw + kw
            y1 = w * sw
            y2 = w * sw + kw
            if mode == 'max':
                output[:, w, h, :] = \
                    np.max(A_prev[:, y1:y2, x1:x2], axis=(1, 2))
            if mode == 'avg':
                output[:, w, h, :] = \
                    np.avg(A_prev[:, y1:y2, x1:x2], axis=(1, 2))
    return output
