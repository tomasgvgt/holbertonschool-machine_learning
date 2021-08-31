Error Analysis

Objectives

    Learn


    What is the confusion matrix?
    What is type I error? type II?
    What is sensitivity? specificity? precision? recall?
    What is an F1 score?
    What is bias? variance?
    What is irreducible error?
    What is Bayes error?
    How can you approximate Bayes error?
    How to calculate bias and variance
    How to create a confusion matrix

Tasks

    0. Create Confusion:

        Write the function def create_confusion_matrix(labels, logits): that creates a confusion matrix:

        labels is a one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point
            m is the number of data points
            classes is the number of classes
        logits is a one-hot numpy.ndarray of shape (m, classes) containing the predicted labels
        Returns: a confusion numpy.ndarray of shape (classes, classes) with row indices representing the correct labels and column indices representing the predicted labels

