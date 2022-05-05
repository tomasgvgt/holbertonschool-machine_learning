#!/usr/bin/env python3
"""Create a pd.DataFrame from a dictionary:"""
import pandas as pd


data = {'First': [0.0, 0.5, 1.0, 1.5],
        'Second': ['one', 'two', 'three', 'four']}
idx = ['A', 'B', 'C', 'D']
df = pd.DataFrame(data, index=idx)
