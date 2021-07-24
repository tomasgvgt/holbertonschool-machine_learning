#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

names = ['Farrah', 'Fred', 'Felicia']
width = 0.5

fig, ax = plt.subplots()


ax.bar(names, fruit[0], width, color='r', label='apples')
ax.bar(names, fruit[1],  width, bottom=fruit[0],
       label='bananas', color='yellow')
ax.bar(names, fruit[2], width,
       bottom=(fruit[0] + fruit[1]), label='oranges', color='orange')
ax.bar(names, fruit[3], width,
       bottom=(fruit[0] + fruit[1] + fruit[2]),
       label='peaches', color='#ffe5b4')

plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.legend()
plt.show()
