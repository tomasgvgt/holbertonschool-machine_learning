#!/usr/bin/env python3
"""Update a variable in place using the Adam optimization algorithm"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm"""
    v_t = beta1 * v + (1 - beta1) * grad
    s_t = beta2 * s + (1 - beta2) * np.power(grad, 2)
    v_hat = v_t / (1 - np.power(beta1, t))
    s_hat = s_t / (1 - np.power(beta2, t))
    var = var - alpha * (v_hat / (np.sqrt(s_hat) + epsilon))
    return var, v_t, s_t
