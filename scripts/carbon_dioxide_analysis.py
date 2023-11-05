import sqlite3

import pandas as pd
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

type_species = "fish "

# read dataset
conn = sqlite3.connect('data.db')
df = pd.read_sql_query(f"""
SELECT carbon_dioxide.Year, biodiversity.\"number of {type_species}species\", carbon_dioxide.\"Annual\" FROM biodiversity, carbon_dioxide where biodiversity.Year = carbon_dioxide.Year
                       """, conn)
conn.close()
# first row is the headers
df = df.iloc[1:]
df.reset_index(drop=True, inplace=True)

# normalize both values
norm = Normalizer()
data = [df[f'number of {type_species}species'], df['Annual']]
data = norm.fit_transform(data)
df[f'number of {type_species}species'] = data[0]
df['Annual'] = data[1]

# print out the correlations
print(pearsonr(data[0], data[1]))

# plot the data
plt.title(f"CO2 over time vs {type_species}species over time")
plt.plot(df['Year'], df[f'number of {type_species}species'], label='Number of species')
plt.plot(df['Year'], df['Annual'], label='CO2')
plt.legend()
plt.show()
