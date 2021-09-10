#!/usr/bin/env python3
"""
Determine if you should stop
gradient descent early
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determine if you should stop
    gradient descent early
    """
    if cost >= opt_cost - threshold:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
    else:
        return False, 0
