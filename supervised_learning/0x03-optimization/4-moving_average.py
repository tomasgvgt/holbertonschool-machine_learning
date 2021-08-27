#!/usr/bin/env python3
"""Calculate the weighted moving average of a data set"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set"""
    moving_average = []
    val = 0
    i = 0
    bias_correction = 1 / ((1 - (beta ** (i + 1))))

    for i in range(len(data)):
        val = (beta * val + (1 - beta) * data[i]) * bias_correction
        moving_average.append(val)
    return moving_average
