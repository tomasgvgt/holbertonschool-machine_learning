#!/usr/bin/env python3
"""
Conduct forward propagation using dropout
"""
import numpy as np


def dropout_forward_prop(X, wheights, L, keep_prob):
    """
    Conducts forward propagation using dropout
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        layer = str(i + 1)
        prev_layer = str(i)
        A_key = 'A' + layer
        d_key = 'D' + layer
        W_key = 'W' + layer
        b_key = 'b' + layer
        Zl = np.matmul(wheights[W_key],
                       cache['A' + prev_layer]) + wheights[b_key]
        if i == L - 1:
            t = np.exp(Zl)
            cache[A_key] = t / np.sum(t, axis=0, keepdims=True)
        else:
            cache[A_key] = np.tanh(Zl)
            cache[d_key] = np.random.binomial(
                1, keep_prob, (cache[A_key].shape[0], cache[A_key].shape[1]))
            cache[A_key] = np.multiply(cache[A_key], cache[d_key])
            cache[A_key] /= keep_prob
    return cache
