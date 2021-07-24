#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.margins(x=0, y=0)
plt.suptitle("Exponential Decay of Radioactive Elements", y=0.93)
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.plot(x, y1, linestyle='dashed', c='r', label="C-14")
plt.plot(x, y2, c='g', label="Ra-226")
plt.legend()
plt.show()
