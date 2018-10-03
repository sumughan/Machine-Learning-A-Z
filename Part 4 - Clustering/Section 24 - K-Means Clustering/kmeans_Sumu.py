# K-Means Algorithm

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters = i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.title('Elbow Chart to get optimal Cluster')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS value')
plt.show()

kmeans=KMeans(n_clusters = 5, init='k-means++', random_state=0)
y=kmeans.fit_predict(X)

plt.scatter(X[y==0,0],X[y==0,1],s=100,color='red',label='Cluster1')
plt.scatter(X[y==1,0],X[y==1,1],s=100,color='blue',label='Cluster2')
plt.scatter(X[y==2,0],X[y==2,1],s=100,color='green',label='Cluster3')
plt.scatter(X[y==3,0],X[y==3,1],s=100,color='cyan',label='Cluster4')
plt.scatter(X[y==4,0],X[y==4,1],s=100,color='magenta',label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='yellow',s=200)
plt.legend()
plt.title('Customer Analysis Cluster')
plt.show()
