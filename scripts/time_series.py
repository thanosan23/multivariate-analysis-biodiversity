from statsmodels.tsa.seasonal import seasonal_decompose
import sqlite3

import pandas as pd
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt

type_species = ""

# read dataset
conn = sqlite3.connect('data.db')
df = pd.read_sql_query(f"""
            SELECT Year, \"number of {type_species}species\" FROM biodiversity
                       """, conn)
conn.close()
# first row is the headers
df = df.iloc[1:]
df.reset_index(drop=True, inplace=True)

df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

decompose_result_mult = seasonal_decompose(df[f"number of {type_species}species"], model="multiplicative")

trend = decompose_result_mult.trend
seasonal = decompose_result_mult.seasonal
residual = decompose_result_mult.resid

decompose_result_mult.plot()
plt.show()
