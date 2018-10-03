# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

from scipy.cluster.hierarchy import dendrogram, linkage

dent=dendrogram(linkage(X,method = 'ward'))

from sklearn.cluster import AgglomerativeClustering

hierarchical=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')

y=hierarchical.fit_predict(X)

plt.scatter(X[y==0,0],X[y==0,1],s=100,color='red',label='Cluster1')
plt.scatter(X[y==1,0],X[y==1,1],s=100,color='blue',label='Cluster2')
plt.scatter(X[y==2,0],X[y==2,1],s=100,color='green',label='Cluster3')
plt.scatter(X[y==3,0],X[y==3,1],s=100,color='cyan',label='Cluster4')
plt.scatter(X[y==4,0],X[y==4,1],s=100,color='magenta',label='Cluster5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()