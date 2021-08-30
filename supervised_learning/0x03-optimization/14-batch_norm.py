#!/usr/bin/env python3
"""
create a batch normalization layer
for a neural network in tensorflow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    create a batch normalization layer
    for a neural network in tensorflow
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initializer)
    z = layer(prev)
    z_mean, z_variance = tf.nn.moments(z, axes=0)
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    z_normalized = tf.nn.batch_normalization(z, z_mean, z_variance,
                                             beta, gamma, 1e-8)
    activated_output = activation(z_normalized)
    return activated_output
