#!/usr/bin/env python3


def poly_derivative(poly):
    derivative = []
    if len(poly) == 0 or not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
    return derivative
