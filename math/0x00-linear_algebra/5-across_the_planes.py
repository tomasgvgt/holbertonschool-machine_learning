#!/usr/bin/env python3
"""Add two 2Dmatrixes element-wise"""


def add_matrices2D(mat1, mat2):
    """Adds two 2Dmatrixes element-wise"""
    mat3 = []
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    for i in range(len(mat1)):
        row3 = []
        for j in range(len(mat1[0])):
            row3.append(mat1[i][j] + mat2[i][j])
        mat3.append(row3)
    return mat3
