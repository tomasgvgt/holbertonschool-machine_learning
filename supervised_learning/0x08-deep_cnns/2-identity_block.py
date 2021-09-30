#!/usr/bin/env python3
"""
Build an identity block as described in
'Deep residual learning for image recognition (2015)'
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep residual learning for image recognition (2015)'

    Arguments:
    - A_prev is the output from the previous layer
    - filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution

    Requisites:
    All convolutions inside the block should be followed by batch
        normalization along the channels axis and a rectified
        linear activation (ReLU), respectively.
    All weights should use he normal initialization

    Returns:
        The activated output of the identity block
    """
    F11, F3, F12 = filters
    kernel_init = K.initializers.he_normal(seed=None)

    lay1 = K.layers.Conv2D(kernel_size=(1, 1), filters=F11,
                           padding='same',
                           kernel_initializer=kernel_init)(A_prev)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(lay1)
    activation1 = K.layers.Activation('relu')(batch_norm1)

    lay2 = K.layers.Conv2D(kernel_size=(3, 3), filters=F3,
                           padding='same',
                           kernel_initializer=kernel_init)(activation1)
    batch_norm2 = K.layers.BatchNormalization(axis=3)(lay2)
    activation2 = K.layers.Activation('relu')(batch_norm2)

    lay3 = K.layers.Conv2D(kernel_size=(1, 1), filters=F12,
                           padding='same',
                           kernel_initializer=kernel_init)(activation2)
    batch_norm3 = K.layers.BatchNormalization(axis=3)(lay3)
    add = K.layers.Add()([batch_norm3, A_prev])
    activation3 = K.layers.Activation('relu')(add)

    return activation3
