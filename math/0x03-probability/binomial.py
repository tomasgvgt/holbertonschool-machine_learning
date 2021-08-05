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
            p = (sum(data) / n) / 100
            p = float(p)
        self.n = n
        self.p = p
