#!/usr/bin/env python3
"""trains a loaded neural network model using mini-batch gradient descent"""
import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def batch(T, n):
    batches = []
    lenght = len(T)
    for ndx in range(0, lenght, n):
        batches.append(T[ndx:min(ndx + n, lenght)])
    return batches


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(session, load_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        for i in range(epochs + 1):
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
            training_loss, training_accuracy = session.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train})
            validation_loss, validation_accuracy = session.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(training_loss))
            print('\tTraining Accuracy: {}'.format(training_accuracy))
            print('\tValidation Cost: {}'.format(validation_loss))
            print('\tValidation Accuracy: {}'.format(validation_accuracy))

            if i < epochs:
                X_batches = batch(X_shuffled, batch_size)
                Y_batches = batch(Y_shuffled, batch_size)
                for j in range(1, len(X_batches) + 1):
                    session.run(train_op, feed_dict={x: X_batches[j - 1],
                                y: Y_batches[j - 1]})
                    training_loss, training_accuracy = session.run(
                        [loss, accuracy], feed_dict={
                            x: X_batches[j - 1], y: Y_batches[j - 1]})

                    if j % 100 == 0:
                        print('\tStep {}:'.format(j))
                        print('\t\tCost: {}'.format(training_loss))
                        print('\t\tAccuracy: {}'.format(training_accuracy))

        return saver.save(session, save_path)
