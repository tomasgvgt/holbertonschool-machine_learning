#!/usr/bin/env python3
"""
Build a NN with the keras library
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a NN with the keras library

    Parameters:
        nx: Number of input features to the network
        layers: List containing the number of nodes
            in each layer of the network
        activations: List containing the activation functions
            used for each layer of the network
        lambtha: L2 regularization parameter
        keep_prob: Probability that a node will be kept for dropout.
    Returns:
        The keras model.
    """
    regularizer = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    outputs = K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=regularizer)(inputs)
    for i in range(1, len(layers)):
        dropout = K.layers.Dropout(rate=(1-keep_prob))(outputs)
        outputs = K.layers.Dense(layers[i], input_shape=(nx,),
                                 activation=activations[i],
                                 kernel_regularizer=regularizer)(dropout)
        model = K.models.Model(inputs=inputs, outputs=outputs)
    return model
