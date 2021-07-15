#!/usr/bin/env python3
"""transpose a 2D matrix"""


def matrix_transpose(matrix):
    """transposes a 2d matrix"""
    transpose = []
    zipped_rows = zip(*matrix)
    transpose = [list(row) for row in zipped_rows]
    return transpose
