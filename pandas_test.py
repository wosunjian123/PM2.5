import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame({'color' : ['white', 'red', 'green', 'red', 'green'],
                'object': ['pen', 'pencil', 'pencil', 'ashtray', 'pen'],
                'price1': [5.56, 4.20, 1.30, 0.56, 2.75],
                'price2': [4.75, 4.12, 1.60, 0.75, 3.15]})
g = df.groupby(['color','object'])

print(g['price1'].sum())