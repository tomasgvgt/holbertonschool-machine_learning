#!/usr/bin/env python3
"""
Create a layer of a newral network
using dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer of neural network using dropout
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    regularizer = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)
    return layer(prev)
