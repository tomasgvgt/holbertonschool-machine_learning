#!/usr/bin/env python3
"""
Build the Resnet50 architecture as described in
'Deep Residual Learning for Image recognition (2015)'
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the Resnet50 architecture as described in
    'Deep Residual Learning for Image recognition (2015)'

    Requisites:
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be
        followed by batch normalization along the channels axis
        and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization

    Returns:
        The Keras model
    """
    kernel_init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))

    lay1 = K.layers.Conv2D(kernel_size=(7, 7), filters=64,
                           padding='same', strides=2,
                           kernel_initializer=kernel_init)(X)
    batch_norm1 = K.layers.BatchNormalization(axis=3)(lay1)
    activation1 = K.layers.Activation('relu')(batch_norm1)
    pool1 = K.layers.MaxPool2D(pool_size=[3, 3], strides=2,
                               padding='same')(activation1)
    lay2 = projection_block(pool1, [64, 64, 256], 1)
    lay3 = identity_block(lay2, [64, 64, 256])
    lay4 = identity_block(lay3, [64, 64, 256])
    lay5 = projection_block(lay4, [128, 128, 512])
    lay6 = identity_block(lay5, [128, 128, 512])
    lay7 = identity_block(lay6, [128, 128, 512])
    lay8 = identity_block(lay7, [128, 128, 512])
    lay9 = projection_block(lay8, [256, 256, 1024])
    lay10 = identity_block(lay9, [256, 256, 1024])
    lay11 = identity_block(lay10, [256, 256, 1024])
    lay12 = identity_block(lay11, [256, 256, 1024])
    lay13 = identity_block(lay12, [256, 256, 1024])
    lay14 = identity_block(lay13, [256, 256, 1024])
    lay15 = projection_block(lay14, [512, 512, 2048])
    lay16 = identity_block(lay15, [512, 512, 2048])
    lay17 = identity_block(lay16, [512, 512, 2048])
    avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7],
                                         padding='same')(lay17)
    Y = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=kernel_init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=Y)
    return model
