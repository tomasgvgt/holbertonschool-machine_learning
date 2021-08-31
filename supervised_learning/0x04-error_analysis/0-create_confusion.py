#!/usr/bin/env python3
"""Create a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion Matrix"""
    confusion_matrix = np.zeros((len(labels[0]), len(labels[0])))
    correct_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    for a, p in zip(correct_classes, pred_classes):
        confusion_matrix[a][p] += 1
    return confusion_matrix
