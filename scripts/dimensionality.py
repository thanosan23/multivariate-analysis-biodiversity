import sqlite3

import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

type_species = ""

# read dataset
conn = sqlite3.connect('data.db')
df = pd.read_sql_query(f"""
SELECT pollution.\"Volatile organic compounds\", biodiversity.\"number of {type_species}species\", carbon_dioxide.\"Annual\" FROM biodiversity, carbon_dioxide, pollution where biodiversity.Year = carbon_dioxide.Year AND carbon_dioxide.Year = pollution.Year
                       """, conn)
conn.close()
# first row is the headers
df = df.iloc[1:]
df.reset_index(drop=True, inplace=True)

# # normalize both values
norm = Normalizer()
data = [df[f'number of {type_species}species'], df['Annual'], abs(df['Volatile organic compounds'])]
data = norm.fit_transform(data)
df[f'number of {type_species}species'] = data[0]
df['Annual'] = data[1]
df['Volatile organic compounds'] = data[2]

X = df[['number of species', 'Annual', 'Volatile organic compounds']]

X_embedded = TSNE(n_components=2, perplexity=10).fit_transform(X)
plt.plot(X_embedded[:, 0], X_embedded[:, 1], 'o')
plt.show()

# clusters = KMeans(n_clusters=2, n_init='auto').fit_predict(X)
# df['cluster'] = clusters.astype(str)
# colors = ['r','b','y','g','c','m']
# for i in range(len(df)):
#     plt.plot(df['number of species'][i], df['Annual'][i], 'o', color=colors[int(df['cluster'][i])])
# plt.show()

# pca = PCA(n_components=2)
# X_embedded = pca.fit_transform(X)
# plt.plot(X_embedded[:, 0], X_embedded[:, 1], 'o')
# plt.show()
