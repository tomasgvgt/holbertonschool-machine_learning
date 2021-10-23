#!/usr/bin/env python3
"""
Calculate the determinant of a maatrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    Arguments:
    - matrix is a list of lists whose determinant should be calculated

    Conditionals:
    If matrix is not a list of lists,
        raise a TypeError with the message matrix must be a list of lists
    If matrix is not square, raise a ValueError
        with the message matrix must be a square matrix
    The list [[]] represents a 0x0 matrix

    Return:
    the determinant of matrix
    """

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError('matrix must be a square matrix')
    if len(matrix) == 1:
        return matrix[0][0]

    deter = 0
    for i, j in enumerate(matrix[0]):
        row = [r for r in matrix[1:]]
        sub_mat = []
        for r in row:
            aux = []
            for c in range(len(matrix)):
                if c != i:
                    aux.append(r[c])
            sub_mat.append(aux)
        deter += (-1) ** i * j * determinant(sub_mat)

    return deter
