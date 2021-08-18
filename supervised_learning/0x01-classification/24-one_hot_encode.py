#!/usr/bin/env python3
"""Convert a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) or len(Y) < 1:
        return None
    if not isinstance(classes, int) or classes < 1 or classes < np.amax(Y):
        return None
    one_hot = np.zeros((classes, Y.size))
    one_hot[Y, np.arange(Y.size)] = 1
    return one_hot
