#!/usr/bin/env python3
"""
Build a modified version of the LeNet-5 architecture
using tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """
    Builds a modified version of the LetNet-5 architecture
    using tensorflow

    Arguments:
    - x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
        images for the network, m is the number of images
    - y is a tf.placeholder of shape (m, 10) containing the one-hot
        abels for the network

    Requisites:
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
        the he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation should use
        the relu activation function

    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization
            (with default hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """

    kernel_init = tf.contrib.layers.variance_scaling_initializer()
    relu_activation = tf.nn.relu

    conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                             activation=relu_activation,
                             kernel_initializer=kernel_init)(x)
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                             activation=relu_activation,
                             kernel_initializer=kernel_init)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)
    flatten = tf.layers.Flatten()(pool2)
    lay3 = tf.layers.Dense(units=120, activation=relu_activation,
                           kernel_initializer=kernel_init)(flatten)
    lay4 = tf.layers.Dense(units=84, activation=relu_activation,
                           kernel_initializer=kernel_init)(lay3)
    output_lay = tf.layers.Dense(units=10,
                                 kernel_initializer=kernel_init)(lay4)
    y_hat = tf.nn.softmax(output_lay)
    loss = tf.losses.softmax_cross_entropy(y, output_lay)
    train = tf.train.AdamOptimizer().minimize(loss)
    equality = tf.equal(tf.argmax(y, axis=1),
                        tf.argmax(output_lay, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return y_hat, train, loss, accuracy
