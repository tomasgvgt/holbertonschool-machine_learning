#!/usr/bin/env python3
"""
Build a modified version on the LeNet-5 architecture
using keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of LeNet-5 architecture

    Arguments:
    X is a K.Input of shape (m, 28, 28, 1) containing the
        input images for the network, m is the number of images

    Requisites:
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize
        their kernels with the he_normal initialization method
    All hidden layers requiring activation should use the
        relu activation function

    Returns:
        a K.Model compiled to use Adam optimization
        (with default hyperparameters) and accuracy metrics
    """

    kernel_init = K.initializers.he_normal(seed=None)
    relu_activation = 'relu'

    conv1 = K.layers.Conv2D(filters=6, kernel_size=5,
                            padding='same', activation=relu_activation,
                            kernel_initializer=kernel_init)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5,
                            padding='valid', activation=relu_activation,
                            kernel_initializer=kernel_init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flatten = K.layers.Flatten()(pool2)
    lay3 = K.layers.Dense(120, activation=relu_activation,
                          kernel_initializer=kernel_init)(flatten)
    lay4 = K.layers.Dense(84, activation=relu_activation,
                          kernel_initializer=kernel_init)(lay3)
    output_lay = K.layers.Dense(10, activation='softmax',
                                kernel_initializer=kernel_init)(lay4)
    model = K.models.Model(X, output_lay)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
