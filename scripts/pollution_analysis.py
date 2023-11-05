import sqlite3

import polars as pl
import pandas as pd
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# read dataset
conn = sqlite3.connect('data.db')
df = pd.read_sql_query("""
SELECT pollution.Year, biodiversity.\"number of species\", pollution.\"Volatile organic compounds\" FROM biodiversity, pollution where biodiversity.Year = pollution.Year
                       """, conn)
conn.close()
# first row is the headers
df = df.iloc[1:]
df.reset_index(drop=True, inplace=True)

# normalize both the number of species and VOC change over time
norm = Normalizer()
# take absolute value of VOC to get the magnitude of change not the direction
data = [df['number of species'], abs(df['Volatile organic compounds'])]
data = norm.fit_transform(data)
df['number of species'] = data[0]
df['Volatile organic compounds'] = data[1]

# print out the correlations
print(pearsonr(data[0], data[1]))

# plot the data
plt.title("Pollution over time vs species over time")
plt.plot(df['Year'], df['number of species'], label='Number of species')
plt.plot(df['Year'], df['Volatile organic compounds'], label='Pollution')
plt.legend()
plt.show()
