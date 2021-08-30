#!/usr/bin/env python3
"""
Model that builds, trains, and saves a neural network model
in tensorflow using Adam optimization,
mini-batch gradient descent, learning rate decay,
and batch normalization
"""
import numpy as np
import tensorflow as tf


def create_placeholders(nx, classes):
    """ returns two placeholders, x and y, for the neural network"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(name='layer', units=n,
                            kernel_initializer=initializer,
                            activation=activation)
    return layer(prev)


def forward_prop(x, layer_sizes=[], activations=[], epsilon=1e-8):
    """creates the forward propagation graph for the neural network"""
    for i in range(len(layer_sizes)):
        if i < len(layer_sizes) - 1:
            layer_output = create_batch_norm_layer(x, layer_sizes[i],
                                                   activations[i])
        else:
            layer_output = create_layer(layer_output, layer_sizes[i],
                                        activations[i])
        x = layer_output
    return(layer_output)


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    prediction = tf.argmax(y_pred, axis=1)
    correct_answer = tf.argmax(y, axis=1)
    equality = tf.equal(prediction, correct_answer)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation of a neural network
    with tensorflow using the Adagrand optimization algorithm
    """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )
    return optimizer.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Create a learning rate decay operation
    in tensorflow using inverse time decay
    """
    decay = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True)
    return decay


def create_batch_norm_layer(prev, n, activation):
    """
    create a batch normalization layer
    for a neural network in tensorflow
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initializer)
    z = layer(prev)
    z_mean, z_variance = tf.nn.moments(z, axes=0)
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    z_normalized = tf.nn.batch_normalization(z, z_mean, z_variance,
                                             beta, gamma, 1e-8)
    activated_output = activation(z_normalized)
    return activated_output


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way"""
    shuffler = np.random.permutation(len(X))
    X_shuffled = X[shuffler]
    Y_shuffled = Y[shuffler]
    return X_shuffled, Y_shuffled


def batch(T, n):
    """Create batches"""
    batches = []
    lenght = len(T)
    for ndx in range(0, lenght, n):
        batches.append(T[ndx:min(ndx + n, lenght)])
    return batches


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Model that builds, trains, and saves a neural network model
    in tensorflow using Adam optimization,
    mini-batch gradient descent, learning rate decay,
    and batch normalization
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    x, y = create_placeholders(nx, classes)
    layer_output = forward_prop(x, layers, activations, epsilon)
    loss = calculate_loss(y, layer_output)
    accuracy = calculate_accuracy(y, layer_output)
    m = Data_train[0].shape[0]
    num_batches = int(m / batch_size) + (m % batch_size > 0)
    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign_add(
        global_step, 1, name='increment_global_step'
    )
    alpha = learning_rate_decay(alpha, decay_rate, global_step, num_batches)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(initializer)
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
                    session.run((increment_global_step, train_op),
                                feed_dict={x: X_batches[j - 1],
                                y: Y_batches[j - 1]})
                    training_loss, training_accuracy = session.run(
                        [loss, accuracy], feed_dict={
                            x: X_batches[j - 1], y: Y_batches[j - 1]})

                    if j % 100 == 0:
                        print('\tStep {}:'.format(j))
                        print('\t\tCost: {}'.format(training_loss))
                        print('\t\tAccuracy: {}'.format(training_accuracy))
        return saver.save(session, save_path)
