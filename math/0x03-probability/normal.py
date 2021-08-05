#!/usr/bin/env python3
"""Represents a normal distribution"""


class Normal:
    """Represents a Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initiates a Noraml (Gaussian or Bell) distribution
        Sets the mean and the standar deviation
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # 𝞵 = sum of all data / number of data points
            mean = sum(data) / len(data)
            # the standar deviation (ơ) is the square root of the variance
            # ơ = √ ((∑ (xi - 𝞵)^2) / N)
            suma = 0
            for i in data:
                suma += (i - mean) ** 2
            variance = suma / len(data)
            stddev = variance ** 0.5
        self.mean = float(mean)
        self.stddev = float(stddev)

    def z_score(self, x):
        """
        calculates the Z score given an x value
        """
        # Z = (x - 𝞵) / ơ
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x value of a given z score
        """
        # x = (Z * ơ) + 𝞵
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calcuates the PDF for a given x
        """
        e = 2.7182818285
        π = 3.1415926536
        # f(x) = (e ** − (x−μ) ** 2 / (2σ ** 2)) / σ * √(2π)
        div = e ** (- ((x - self.mean) ** 2) / (2 * self.stddev ** 2))
        divisor = self.stddev * ((2 * π) ** 0.5)
        return div / divisor
