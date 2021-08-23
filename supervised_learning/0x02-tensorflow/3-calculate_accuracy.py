#!/usr/bin/env python3
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    prediction = tf.argmax(y_pred, axis=1)
    correct_answer = tf.argmax(y, axis=1)
    equality = tf.equal(prediction, correct_answer)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
