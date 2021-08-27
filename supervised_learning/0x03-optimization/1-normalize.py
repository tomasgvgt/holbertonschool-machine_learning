#!/usr/bin/env python3
"""normalize (standardizes) a matrix"""


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    X_normalized = (X - m) / s
    return X_normalized
