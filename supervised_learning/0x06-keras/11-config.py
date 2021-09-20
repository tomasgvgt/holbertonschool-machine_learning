#!/usr/bin/env python3
"""
Save and load configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model configuration in JSON
    Arguments:
        network: model whose configuration should be saved
        filename: path of the file that the configuration should be saved to
    Return:
        None
    """

    json_network = network.to_json()
    with open(filename, 'w') as f:
        f.write(json_network)
    return None


def load_config(filename):
    """
    loads a model with a specific configuration
    Arguments:
        filename: the path of the file containing the model’s
        configuration in JSON format
        Returns: the loaded model
    Returns:
        None
    """

    with open(filename, 'r') as f:
        json_network = K.models.model_from_json(f.read())
    return json_network
