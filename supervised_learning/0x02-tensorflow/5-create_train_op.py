#!/usr/bin/env python3
"""creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """operation that trains the network using gradient descent"""
    train = tf.train.GradientDescentOptimizer(
        learning_rate=alpha).minimize(loss=loss)
    return train
