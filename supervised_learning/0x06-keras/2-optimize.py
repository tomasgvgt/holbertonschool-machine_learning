#!/usr/bin/env python3
"""
Set up Adam optimization for a keras model with
categorical crossentropy loss and accuracy metrics
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with
    categorical crossentropy loss and accuracy metrics

    Arguments:
    network: model to optimize
    alpha: learning rate
    beta1: first Adam optimization parameter
    beta2: second Adam optimization parameter

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy', optimizer=optimizer,
                    metrics=['accuracy'])
    return None
