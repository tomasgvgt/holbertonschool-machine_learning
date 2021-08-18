#!/usr/bin/env python3
"""Convert a numeric label vector into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix"""
    one_hot = np.zeros((classes, Y.max()+1))
    one_hot[Y, np.arange(Y.size)] = 1
    return one_hot
