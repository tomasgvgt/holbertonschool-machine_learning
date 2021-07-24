#!/usr/bin/env python3
"""Calculate a sum"""


def summation_i_squared(n):
    """Calculates a sum"""
    summation = 0
    for i in range(1, n + 1):
        summation += i ** 2
    return summation
