#!/usr/bin/env python3
"""
Calculate the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    """
    sensitivity = np.zeros(len(confusion),)
    for i in range(len(confusion)):
        # Sensitivity  = TruePositive/ActualClass
        actual_class = np.sum(confusion[i])
        true_positive = confusion[i][i]
        sensitivity[i] = true_positive / actual_class
    return sensitivity
