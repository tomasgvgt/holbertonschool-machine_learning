#!/usr/bin/env python3
"""
BUild a dense block as described in
'Densely connected convolutional Networks'
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    BUilds a dense block as described in
    'Densely connected convolutional Networks'

    Arguments:
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block

    Requisites:
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    All convolutions should be preceded by Batch Normalization and a
        rectified linear activation (ReLU), respectively

    Returns:
    The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """

    kernel_init = K.initializers.he_normal(seed=None)

    for lay in range(layers):
        batch_norm1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(batch_norm1)
        conv1 = K.layers.Conv2D(kernel_size=1, filters=4*growth_rate,
                                padding='same',
                                kernel_initializer=kernel_init)(activation1)

        batch_norm2 = K.layers.BatchNormalization()(conv1)
        activation2 = K.layers.Activation('relu')(batch_norm2)
        conv2 = K.layers.Conv2D(kernel_size=3, filters=growth_rate,
                                padding='same',
                                kernel_initializer=kernel_init)(activation2)

        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
