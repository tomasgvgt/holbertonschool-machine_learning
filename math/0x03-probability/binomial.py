#!/usr/bin/env python3
"""Represents a binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor
        Sets n and p
        p is the probability of success
        n is the number of bernoulli trials
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            # variance = ∑ (xi - 𝞵)^2) / N
            suma = 0
            for i in data:
                suma += (i - mean) ** 2
            variance = suma / len(data)
            # mean = np
            # variance = npq
            # q = npq / np
            # So having the variance and the mean we can calculate q
            q = variance / mean
            # p + q = 1
            # p = 1 - q
            p = 1 - q
            # if mean = np
            # then n = mean / p
            n = round(mean / p)
            p = mean / n
        self.n = int(round(n))
        self.p = float(p)

    @staticmethod
    def factorial(n):
        if n > 1:
            return n * Binomial.factorial(n - 1)
        else:
            return 1

    def pmf(self, k):
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        # f (k; n; p) = (n! / (k! * (n - x)!)) * (p ** k * (1 - p) ** (n - k))
        return (Binomial.factorial(self.n) / (
            Binomial.factorial(k) * Binomial.factorial(self.n - k))) * (
            (self.p ** k) * ((1 - self.p) ** (self.n - k)))
