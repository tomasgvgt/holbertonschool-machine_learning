#!/usr/bin/env python3
"""Save and load model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model

    Arguments:
    network: model to save
    filename: path of the file that
    the model should be saved to
    Returns: None
    """

    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model

    filename is the path of the file that
    the model should be loaded from
    Returns: the loaded model
    """
    model = K.models.load_model(filename)
    return model
