#!/usr/bin/env python3
"""
Make predictions using a NN
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a NN
    Arguments:
        network: network model to make the prediction with
        data: input data to make the prediction with
        verbose: boolean that determines if output should
        be printed during the prediction process
    Returns:
        THe prediciton of the data
    """
    prediction = network.predict(x=data, verbose=verbose)
    return prediction
