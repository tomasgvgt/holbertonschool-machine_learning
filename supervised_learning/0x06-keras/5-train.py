#!/usr/bin/env python3
"""
Based on 4-train.py, update the function train_model
to also analyze validation data
"""


def train_model(network, data, labels, batch_size,
                validation_data=None, epochs,
                verbose=True, shuffle=False,
                ):
    """
    Trains a model using mini batch gradient descent

    Arguments:
        network: Model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data
        batch_size: Size of the batch used for mini-batch gradient descent
        epochs: Number of passes through data for mini-batch gradient descent
        verbose: Boolean that determines if output should be
            printed during training
        shuffle: Boolean that determines whether to shuffle
            the batches every epoch.
            Normally, it is a good idea to shuffle, but for reproducibility,
            we have chosen to set the default to False.
        validation_data: data to validate the model with, if not None
    Returns:
        History object generated after training the model
    """

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data)
    return history
