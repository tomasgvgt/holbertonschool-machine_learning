#!/usr/bin/env python3
"""
Save and load the model weights
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves the model weights
    Arguments:
        network: model whose weights should be saved
        filename: path of the file that the weights should be saved to
        save_format: format in which the weights should be saved
    Return:
        None
    """

    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """
    Loads the models weights
    Arguments:
        network: model to which the weights should be loaded
        filename: path of the file that the weights should be loaded from
    Return:
        None
    """
    network.load_weights(filename)
    return None
