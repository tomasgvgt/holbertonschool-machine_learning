#!/usr/bin/env python3
"""
Create the training operation for a neural network
in tensorflow using the RMSProp optimization algorithm
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm
    """
    optimizer = tf.train.RMSPropOptimizer(alpha, epsilon=epsilon, decay=beta2)
    return optimizer.minimize(loss)
