#!/usr/bin/env python3
"""Concatenate two lists"""


def cat_arrays(arr1, arr2):
    """Concatenates two lists"""
    arr3 = arr1.copy()
    for element in arr2:
        arr3.append(element)
    return arr3
