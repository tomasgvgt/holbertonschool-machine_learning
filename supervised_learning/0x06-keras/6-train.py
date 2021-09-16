#!/usr/bin/env python3
"""
Based on 5-train.py, update the function train_model
to also train the model using early stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
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
        early_stopping: Boolean that indicates
            whether early stopping should be used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss
        patience: Patience used for early stopping

    Returns:
        History object generated after training the model
    """
    callbacks = []
    if early_stopping and validation_data:
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose, shuffle=shuffle,
                              validation_data=validation_data,
                              callbacks=callbacks)
    return history
