#!/usr/bin/env python3
"""COncatenate two 2d matrixes in a specific axes"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2d matrixes in a specific axes"""
    mat3 = []
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        for row in mat1:
            mat3.append(row.copy())
        for row in mat2:
            mat3.append(row.copy())

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            row3 = mat1[i].copy()
            row3.extend(mat2[i])
            mat3.append(row3)
    return mat3
