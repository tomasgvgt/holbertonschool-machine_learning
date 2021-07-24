#!/usr/bin/env python3
"""Calculate a sum"""


def summation_i_squared(n):
    """Calculates a sum"""
    summation = 0
    if not isinstance(n, int):
        return None
    for i in range(n):
        summation += i ** 2
    summation += n **2 
    return summation
