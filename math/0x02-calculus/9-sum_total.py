#!/usr/bin/env python3
"""Calculate a sum"""


def summation_i_squared(n):
    """Calculates a sum"""
    if not isinstance(n, (int, float)):
        return None
    if n == 1:
        return 1
    else:
        return (n ** 2 + summation_i_squared(n - 1))
