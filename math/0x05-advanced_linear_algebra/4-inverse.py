#!/usr/bin/env python3
"""
Calculate the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    Arguments:
    matrix is a list of lists whose minor matrix should be calculated

    Requisites:
    If matrix is not a list of lists, raise a TypeError
        with the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
        with the message matrix must be a non-empty square matrix
    Returns:
    the minor matrix of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1]]

    minors = [[0 for j in range(len(matrix))] for i in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            sub_matrix = [[matrix[r][c] for c in range(len(matrix)) if c != j]
                          for r in range(len(matrix)) if r != i]
            minors[i][j] = determinant(sub_matrix)

    return minors


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """
    if len(matrix) == 1:
        return matrix[0][0]

    deter = 0
    for i in range(len(matrix)):
        sub_matrix = [matrix[r][1:] for r in range(len(matrix)) if r != i]
        deter += (-1) ** (i) * matrix[i][0] * determinant(sub_matrix)

    return deter


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix:

    Arguments:
    matrix is a list of lists whose cofactor matrix should be calculated

    Requisites:
    If matrix is not a list of lists, raise a TypeError with the
        message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with
        the message matrix must be a non-empty square matrix
    Returns:
        the cofactor matrix of matrix
    """
    cof = minor(matrix)
    for i in range(len(cof)):
        for j in range(len(cof[0])):
            cof[i][j] *= (-1) ** (i + j)
    return cof


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix:
    Arguments:
    matrix is a list of lists whose adjugate matrix should be calculated

    Requisites:
    If matrix is not a list of lists, raise a TypeError
        with the message matrix must be a list of lists
    If matrix is not square or is empty
        raise a ValueError with the message matrix must be a
        non-empty square matrix
    Returns:
    the adjugate matrix of matrix
    """
    cof = cofactor(matrix)
    adju = [[cof[j][i] for j in range(len(cof))] for i in range(len(cof))]
    return adju


def inverse(matrix):
    """
    calculates the inverse of a matrix
    Arguments:
    matrix is a list of lists whose inverse should be calculated

    Requisites:
    If matrix is not a list of lists, raise a TypeError with the
        message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with
        the message matrix must be a non-empty square matrix

    Returns: the inverse of matrix, or None if matrix is singular
    """
    det = determinant(matrix)
    adju = adjugate(matrix)
    if det is 0:
        return None
    inv = [[i / det for i in row] for row in adju]
    return inv
