#!/usr/bin/env/python3


def matrix_transpose(matrix):
    """transposes a 2d matrix"""
    transpose = []
    
    zipped_rows = zip(*matrix)
    transpose = [list(row) for row in zipped_rows]
    return transpose
