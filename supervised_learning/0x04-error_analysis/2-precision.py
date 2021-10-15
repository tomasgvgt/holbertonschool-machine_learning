#!/usr/bin/env python3
"""
Calculate the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    """
    precision = np.zeros(len(confusion),)
    all_predicted_class = confusion.sum(axis=0)
    for i in range(len(confusion)):
        # precision = TruePositive / PredictedClass
        true_positive = confusion[i][i]
        predicted_class = all_predicted_class[i]
        precision[i] = true_positive / predicted_class
    return precision
