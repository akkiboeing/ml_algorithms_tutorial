# Principal Component Analysis
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

print(type(cancer))

print(cancer.keys())

print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
print(df.head())
print(cancer['target'])
print(cancer['target_names'])

scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df) 
pca = PCA(n_components = 2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)

plt.figure(figsize=(10,6))
plt.scatter(x_pca[:,0], x_pca[:,1], c = cancer['target'], cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

print(pca.components_)
df_bc = pd.DataFrame(pca.components_, columns = cancer['feature_names'])
print(df_bc)

plt.figure(figsize =(12,6))
sns.heatmap(df_bc, cmap = 'plasma')
plt.show()