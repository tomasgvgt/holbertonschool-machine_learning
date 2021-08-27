#!/usr/bin/env python3
"""Calculate the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    moving_average = []
    val = 0

    for i in range(len(data)):
        val = (beta * val + (1 - beta) * data[i])
        val_bias_corrected = val / (1 - (beta ** (i + 1)))
        moving_average.append(val_bias_corrected)
    return moving_average
