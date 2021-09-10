#!/usr/bin/env python3
"""
Create a tensorflow layer that includes
l2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes 2l regularization
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(name='layer', units=n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)
    return layer(prev)
