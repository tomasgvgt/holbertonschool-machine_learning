#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y, 'b')
plt.yscale('log')
plt.margins(x=0)
plt.suptitle("Exponential Decay of C-14", y=0.93)
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.show()
