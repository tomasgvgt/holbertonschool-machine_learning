#!/usr/bin/env python3
"""
Calculate the cost of a neural network with L2regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, wheights, L, m):
    """
    Calculates the cost of a neural network with L2regularization
    """
    wheights_sum = 0
    for i in range(1, L + 1):
        k = "W" + str(i)
        wheights_sum += np.linalg.norm(wheights[k])
    l2_cost = cost + lambtha * wheights_sum / (2 * m)
    return l2_cost
