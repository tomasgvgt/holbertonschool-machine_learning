#!/usr/bin/env python3


class Poisson:
    """Represents a poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            n = len(data)
            k = 0
            for i in data:
                k += i
            lambtha = k / n
        self.lambtha = float(lambtha)

    @staticmethod
    def factorial(n):
        """Computes the factorial of a number"""
        if n > 1:
            return n * Poisson.factorial(n - 1)
        else:
            return 1

    def pmf(self, k):
        """Probability Mass Function"""
        # P (X = k) = (ℷ ^k * e ^ - ℷ) / k!
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        lamb = self.lambtha
        e = 2.7182818285
        return lamb ** k * e ** - lamb / Poisson.factorial(k)

    def cdf(self, k):
        """Comulative Mass Function"""
        # F(x) = ∑ from i=0 to x pmf(i)
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        suma = 0.0
        for i in range(k + 1):
            suma += self.pmf(i)
        return suma
