#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """adds two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    addition = arr1.copy()
    for i in range(len(arr2)):
        addition[i] += arr2[i]
    return addition
