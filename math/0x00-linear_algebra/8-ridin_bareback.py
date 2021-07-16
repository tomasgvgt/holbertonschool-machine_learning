#!/usr/bin/env python3
"""Perform a matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs a matrix multiplication"""
    mat3 = []
    if len(mat1[0]) != len(mat2):
        return None
    for i in range(len(mat1)):
        row3 = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(len(mat2)):
                sum += (mat1[i][k] * mat2[k][j])
            row3.append(sum)
        mat3.append(row3)
    return mat3
