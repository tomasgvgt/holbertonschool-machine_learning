#!/usr/bin/env python3
"""Sort the pd.DataFrame by the High price in descending order"""
import pandas as pd
from_file = __import__('2-from_file').from_file


df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_values(by=['High'], ascending=False)

print(df.head())
