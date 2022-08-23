# -*- coding: utf-8 -*-
"""Kmean clustering_GRAFICOS_materna.ipynb"""

from google.colab import drive
drive.mount('/gdrive')

from sklearn.datasets import make_blobs
import pandas as pd

import pandas as pd
df = pd.read_csv('/gdrive/My Drive/Andres Paper/trace-info-o1.1.csv',delimiter=';')
df.head(5)

del df["Tracefile"]
del df["TraceLog"]

"""Correlation"""

import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5,4))         # Sample figsize in inches

sns.heatmap(df.corr(),cmap="Greens")

"""### CPU"""

df_seleccionado = df[['avgCPU MHZ', 'maxCPU MHZ']]

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_seleccionado)
visualizer.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_seleccionado)

CPU_labels = kmeans.labels_  # same as kmeans.predict(df)

# Do the scatter plot and see that clusters are evident
#
plt.scatter(df['avgCPU MHZ'], df['maxCPU MHZ'],
             color='white', marker='o', edgecolor='red', s=50)
plt.grid()
plt.tight_layout()
plt.show()

"""### RAM"""

df_RAM = df[['avgRAM MB', 'maxRAM MB']]

from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_RAM)
visualizer.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_RAM)

RAM_labels = kmeans.labels_  # same as kmeans.predict(df)

plt.scatter(df['avgRAM MB'], df['maxRAM MB'],
             color='white', marker='o', edgecolor='red', s=50)
plt.grid()
plt.tight_layout()
plt.show()

plt.plot(RAM_labels)

CPU_labels

RAM_labels

from matplotlib.pyplot import figure

figure(figsize=(18, 6), dpi=80)

plt.plot(RAM_labels[0:100])
plt.plot(CPU_labels[0:100])

import numpy as np
comparison_column = np.where(RAM_labels == CPU_labels, True, False)

comparison_column

(unique, counts) = np.unique(comparison_column, return_counts=True)

frequencies = np.asarray((unique, counts)).T

frequencies

df['CPU_labels'] = CPU_labels

df['RAM_labels'] = RAM_labels

df['comparison_column'] = comparison_column

df

#df.to_csv('trace-info-o1.1_Kmeans-LABELED.csv')

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=400)#standar: perplexity=40
tsne_results = tsne.fit_transform(df_seleccionado)

solo2df['CPU_labels']=CPU_labels

solo2df=solo2df.rename(columns={0: "X", 1: "Y"})

solo2df

import plotly.express as px

from pylab import rcParams

rcParams['figure.figsize'] = 8, 8

# Plot color
groups = solo2df.groupby('CPU_labels')
solo2df["CPU_labels"] = solo2df["CPU_labels"].astype(str)

# Plot
fig=px.scatter(solo2df, x="X", y="Y",color='CPU_labels',width=800, height=400)
fig.show()


"""RAM

---
"""

tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=400)#standar: perplexity=40
tsne_results = tsne.fit_transform(df_RAM)

solo2df=pd.DataFrame(tsne_results)

solo2df['RAM_labels']=RAM_labels

solo2df=solo2df.rename(columns={0: "X", 1: "Y"})

# Plot color
solo2df["RAM_labels"] = solo2df["RAM_labels"].astype(str)

# Plot
fig=px.scatter(solo2df, x="X", y="Y",color='RAM_labels',width=800, height=400)
fig.show()
