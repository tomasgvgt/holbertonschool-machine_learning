#!/usr/bin/env python3
"""
Build the DenseNet-121 architecture as described in
'Densely connected concolutional networks'
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    'Densely connected concolutional networks'

    Arguments:
    growth_rate is the growth rate
    compression is the compression factor

    Requisites:
    You can assume the input data will have shape (224, 224, 3)
    All convolutions should be preceded by Batch Normalization
        and a rectified linear activation (ReLU), respectively
    All weights should use he normal initialization

    Returns:
    the keras model
    """

    kernel_init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))

    batch_norm = K.layers.BatchNormalization(axis=3)(X)
    activation = K.layers.Activation('relu')(batch_norm)

    conv = K.layers.Conv2D(kernel_size=7, filters=2*growth_rate,
                           strides=2, padding='same',
                           kernel_initializer=kernel_init)(activation)
    pool = K.layers.MaxPool2D(pool_size=[3, 3], strides=2,
                              padding='same')(conv)

    lay1, num_fil1 = dense_block(pool, 2*growth_rate, growth_rate, 6)
    lay2, num_fil2 = transition_layer(lay1, num_fil1, compression)
    lay3, num_fil3 = dense_block(lay2, num_fil2, growth_rate, 12)
    lay4, num_fil4 = transition_layer(lay3, num_fil3, compression)
    lay5, num_fil5 = dense_block(lay4, num_fil4, growth_rate, 24)
    lay6, num_fil6 = transition_layer(lay5, num_fil5, compression)
    lay7, num_fil7 = dense_block(lay6, num_fil6, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7], strides=7,
                                         padding='same')(lay7)

    Y = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel_init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=Y)

    return model
