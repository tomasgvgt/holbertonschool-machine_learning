#!/usr/bin/env python3
"""Create the forward propagation graph for the neural network"""
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network"""
    create_layer = __import__('1-create_layer').create_layer
    for i in range(len(layer_sizes)):
        if i == 0:
            layer_output = create_layer(x, layer_sizes[i], activations[i])
        else:
            layer_output = create_layer(layer_output, layer_sizes[i],
                                        activations[i])
    return(layer_output)
