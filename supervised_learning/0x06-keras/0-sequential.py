#!/usr/bin/env python3
"""
Build a NN with keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the keras library

    Arguments:
        nx: Number of input features to the network
        layers: List containing the number of nodes
            in each layer of the network
        activations: List containing the activation
            -functions used for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: Probability that a node will be kept for dropout
    Returns:
        The Keras model
    """
    regularizer = K.regularizers.l2(lambtha)
    model = K.Sequential(
        [
            K.layers.Dense(layers[0], input_shape=(nx,),
                           activation=activations[0],
                           kernel_regularizer=regularizer)
        ])
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i],
                  activation=activations[i],
                  kernel_regularizer=regularizer))
        model.add(K.layers.Dropout(rate=(1 - keep_prob)))
    return model
