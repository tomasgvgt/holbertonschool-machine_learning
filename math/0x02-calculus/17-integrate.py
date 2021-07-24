#!/usr/bin/env python3


def poly_integral(poly, C=0):
    integral = []
    num = 0
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if any(not isinstance(i, (int, float)) for i in poly):
        return None
    if not isinstance(C, int):
        return None
    for i in range(len(poly) + 1):
        if i == 0:
            integral.append(C)
        else:
            num = poly[i - 1]/i
            if num % 1 == 0:
                num = int(num)
            integral.append(num)
    return integral
