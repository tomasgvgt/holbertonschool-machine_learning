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
    true_positive = np.diag(confusion)
    total_array = np.sum(confusion)
    false_positive = all_predicted - true_positive
    actual_negative = total_array - all_real
    true_negative = actual_negative - false_positive
    return true_negative / actual_negative
