#!/usr/bin/env python3
"""
Build an inception block as described in
'Going deeper with convolutions (2014)'
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block.

    Arguments:

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP.
        F1 is the number of filters in the 1x1 convolution
        F3R is the number of filters in the 1x1 convolution
            before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R is the number of filters in the 1x1 convolution
            before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP is the number of filters in the 1x1 convolution
            after the max pooling

    Requisites:
    All convolutions inside the inception block should use
    a rectified linear activation (ReLU)

    Returns:
    The concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters
    kernel_init = K.initializers.he_normal(seed=None)

    scale1_1x1 = K.layers.Conv2D(filters=F1, kernel_size=1,
                                 padding='same', activation='relu',
                                 kernel_initializer=kernel_init)(A_prev)
    scale2_1x1 = K.layers.Conv2D(filters=F3R, kernel_size=1,
                                 padding='same', activation='relu',
                                 kernel_initializer=kernel_init)(A_prev)
    scale2_3x3 = K.layers.Conv2D(filters=F3, kernel_size=3,
                                 padding='same', activation='relu',
                                 kernel_initializer=kernel_init)(scale2_1x1)
    scale3_1x1 = K.layers.Conv2D(filters=F5R, kernel_size=1,
                                 padding='same', activation='relu',
                                 kernel_initializer=kernel_init)(A_prev)
    scale3_5x5 = K.layers.Conv2D(filters=F5, kernel_size=5,
                                 padding='same', activation='relu',
                                 kernel_initializer=kernel_init)(scale3_1x1)
    pool_3x3 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=1,
                                     padding='same')(A_prev)
    pool_1x1conv = K.layers.Conv2D(filters=FPP, kernel_size=1,
                                   padding='same', activation='relu',
                                   kernel_initializer=kernel_init)(pool_3x3)
    block = K.layers.concatenate([scale1_1x1, scale2_3x3,
                                  scale3_5x5, pool_1x1conv])
    return block
