#!/usr/bin/env python3
"""Representas an exponential distribution"""


class Exponential:
    """Represents an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """
        Initializes the exponential distribution
        setting lambtha according to the given data

        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            miu = sum(data) / len(data)
            lambtha = 1 / miu
        self.lambtha = float(lambtha)

    def pdf(self, x):
        """
        Calculates the value of PDF for a given time period
        """
        # f (x; ℷ) = ℷ * e^(-ℷ * x)
        if x < 0:
            return 0
        lamb = self.lambtha
        e = 2.7182818285
        return lamb * e ** (- lamb * x)

    def cdf(self, x):
        """
        Calculates the value of CDF for a given time period
        """
        # F(x;  ℷ) = 1 - e **(-ℷ * x)
        if x < 0:
            return 0
        lamb = self.lambtha
        e = 2.7182818285
        return 1 - e ** (-lamb * x)
