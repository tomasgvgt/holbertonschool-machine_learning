#!/usr/bin/env python3
"""
Train the model with learning rate decay
save the best iteration of the model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Train the model with learning rate decay.

    Arguments:
        learning_rate_decay: is a boolean that indicates whether
            learning rate decay should be used
            learning rate decay should only be performed if
            validation_data exists
            the decay should be performed using inverse time decay
            the learning rate should decay in a stepwise fashion after
            each epoch
            each time the learning rate updates, Keras should
            print a message
        alpha: initial learning rate
        decay_rate: decay rate
    """
    callbacks = []

    def learning_rate_decay(epoch):
        """Calculates time decay for an epoch"""
        lrate = alpha / (1 + decay_rate * epoch)
        return lrate

    if learning_rate_decay and validation_data:
        lrd = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                verbose=1)
        callbacks.append(lrd)

    if early_stopping and validation_data:
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)

    if filepath:
        checkpoint = K.callbacks.ModelCheckpoint(filepath,
                                                 save_best_only=save_best,
                                                 monitor='val_loss',
                                                 mode='min')
        callbacks.append(checkpoint)

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
