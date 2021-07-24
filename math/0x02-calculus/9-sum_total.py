#!/usr/bin/env python3


def summation_i_squared(n):
    summation = 0
    if not isinstance(n, int):
        return None
    for i in range(n + 1):
        summation += i ** 2
    return summation
