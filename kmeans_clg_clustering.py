import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix

clg = pd.read_csv('College_Data', index_col = 0)

print(clg.head())
print(clg.info())
print(clg.describe())

sns.set_style('whitegrid')
sns.lmplot('Room.Board', 'Grad.Rate', data=clg, hue='Private', palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()

sns.lmplot('Outstate', 'F.Undergrad', data = clg, hue = 'Private', height = 6, aspect = 1, palette = 'coolwarm', fit_reg = False)

g = sns.FacetGrid(clg ,hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
plt.show()

g = sns.FacetGrid(clg ,hue="Private", palette='coolwarm', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()

print(clg[clg['Grad.Rate']>100])
clg['Grad.Rate']['Cazenovia College'] = 100
print(clg[clg['Grad.Rate']>100])

g = sns.FacetGrid(clg,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
plt.show()

kmean = KMeans(n_clusters = 2)
kmean.fit(clg.drop('Private', axis = 1))
print(kmean.cluster_centers_)

def change(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

clg['Cluster'] = clg['Private'].apply(change)
print(clg.head())

print(confusion_matrix(clg['Cluster'],kmean.labels_))
print(classification_report(clg['Cluster'],kmean.labels_))