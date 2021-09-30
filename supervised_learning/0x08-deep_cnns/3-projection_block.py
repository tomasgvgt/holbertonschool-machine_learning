#!/usr/bin/env python3
"""
Build the projection block as described in
'Deep Residual Learning For image recognition (2015)'
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a Projection block

    Arguments:

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection
    s is the stride of the first convolution in both the main
        path and the shortcut connection

    Requisites:
    All convolutions inside the block should be followed
        by batch normalization along the channels axis and
        a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization

    Returns:
        The activated output of the projection block
    """

    F11, F3, F12 = filters
    kernel_init = K.initializers.he_normal(seed=None)

    lay1 = K.layers.Conv2D(kernel_size=(1, 1), filters=F11,
                           padding='same', strides=s,
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

    lay4 = K.layers.Conv2D(kernel_size=(1, 1), filters=F12,
                           padding='same', strides=s,
                           kernel_initializer=kernel_init)(A_prev)
    batch_norm4 = K.layers.BatchNormalization(axis=3)(lay4)

    add = K.layers.Add()([batch_norm3, batch_norm4])
    activation3 = K.layers.Activation('relu')(add)

    return activation3
