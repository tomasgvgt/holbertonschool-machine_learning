#!/usr/bin/env python3
"""Calcualte a derivative"""

def poly_derivative(poly):
    """Calculates a derivative"""
    derivative = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if any(not isinstance(i, int) for i in poly):
        return None
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
    return derivative
