#!/usr/bin/env python3
"""Create a leayer of the neural network"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(name='layer', units=n,
                            kernel_initializer=initializer,
                            activation=activation)
    return layer(prev)
