#!/usr/bin/env python3
"""
Calculate the specificity for each class in a confussion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculate the specificity (true negative rate)
    for each class in a confussion matrix
    """
    # specificity = TrueNegative / actualNo
    # specificity = TrueNegative / TrueNegative + FalsePositive
    specificity = np.zeros(len(confusion),)
    all_predicted = np.sum(confusion, axis=0)
    all_real = np.sum(confusion, axis=1)
    total_array = np.sum(confusion)
    total = np.sum(total_array)
    for i in range(len(confusion)):
        false_positive = all_predicted[i] - confusion[i][i]
        true_negative = total - all_real[i] - confusion[i][i]
        specificity[i] = true_negative / (true_negative + false_positive)
    return specificity
