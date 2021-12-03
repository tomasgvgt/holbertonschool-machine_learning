#!/usr/bin/env python3
"""
Class GaussianProcess, that represents a noiseless 1D
gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    Class GaussianProcess
    that represents a noiseless 1D gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
        X_init is a numpy.ndarray of shape (t, 1) representing
            the inputs already sampled with the black-box function
        Y_init is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init
        t is the number of initial samples
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output
            of the black-box function
        Sets the public instance attributes X, Y, l, and sigma_f corresponding
            to the respective constructor inputs
        Sets the public instance attribute K, representing the current
            covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel
        matrix between two matrices

        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)
        the kernel should use the Radial Basis Function (RBF)
        Returns:
            the covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """
        sqdist = (np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
                  - 2 * np.dot(X1, X2.T))
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a gaussian
        process.


        X_s is a numpy.ndarray of shape (s, 1) containing all of the points
        whose mean and standard deviation should be calculated
            s is the number of sample points
        Returns: mu, sigma
            mu is a numpy.ndarray of shape (s,) containing the mean for
                each point in X_s, respectively
            sigma is a numpy.ndarray of shape (s,) containing the variance
                for each point in X_s, respectively
        """
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)

        return mu_s.reshape(-1), sigma
