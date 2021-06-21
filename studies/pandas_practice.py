import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# print(os.cwd())

df = pd.read_csv(r'..\data\concrete_data.csv')

pd.set_option('display.max_columns',8)
pd.set_option('display.max_rows',1030)

print(df.shape)
print(df.info())
print(df.head())


