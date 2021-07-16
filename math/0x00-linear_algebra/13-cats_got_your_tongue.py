#!/usr/bin/env python3
"""Concatenate two np.ndarrays in a sepcific matrix"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates two np.ndarrays in a sepcific matrix"""
    return np.concatenate((mat1, mat2), axis)
