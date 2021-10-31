#!/usr/bin/env python3
"""
Represent a Multivariable Normal Distribution
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariable Normal Distribution
    """
    def __init__(self, data):
        """
        Class constructor.
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError
            with the message data must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError
            with the message data must contain multiple data points
        """
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.dot((data - self.mean), (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the pdf of a datapoint
        x is a numpy.ndarray of shape (d, 1) containing the data
            point whose PDF should be calculated
        d is the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray, raise a TypeError
            with the message x must be a numpy.ndarray
        If x is not of shape (d, 1), raise a ValueError
            with the message x must have the shape ({d}, 1)
        Returns the value of the PDF
        """
        d = self.cov.shape[0]
        if type(x) != np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        i_cov = np.linalg.inv(self.cov)
        det_cov = np.linalg.det(self.cov)
        x = x - self.mean
        y = np.matmul(np.matmul(x.T, i_cov), x) * (-0.5)
        pdf = np.exp(y[0][0]) / np.sqrt(((2 * np.pi) ** d) * det_cov)
        return pdf
