#!/usr/bin/env python3
"""Calculate the shape of a matrix"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix"""
    shape = []
    for i in range(999999999999):
        if isinstance(matrix, list):
            shape.append(len(matrix))
            matrix = matrix[0]
        else:
            return shape
        i += i
