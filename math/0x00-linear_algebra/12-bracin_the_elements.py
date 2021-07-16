#!/usr/bin/env python3
"""performs element-wise addition, subtraction,
  multiplication, and division to np.ndarrays"""


def np_elementwise(mat1, mat2):
    """performs element-wise addition, subtraction,
    multiplication, and division to np.ndarrays"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
