REGULARIZATION

Objectives:
    Learn:

        What is regularization? What is its purpose?
        What is are L1 and L2 regularization? What is the difference between the two methods?
        What is dropout?
        What is early stopping?
        What is data augmentation?
        How do you implement the above regularization methods in Numpy? Tensorflow?
        What are the pros and cons of the above regularization methods?

    Tasks:

        0. L2 Regularization Cost
            Write a function def l2_reg_cost(cost, lambtha, weights, L, m): that calculates the cost of a neural network with L2 regularization:

                cost is the cost of the network without L2 regularization
                lambtha is the regularization parameter
                weights is a dictionary of the weights and biases (numpy.ndarrays) of the neural network
                L is the number of layers in the neural network
                m is the number of data points used
                Returns: the cost of the network accounting for L2 regularization
