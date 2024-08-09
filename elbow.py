import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('wine-clustering.csv')

# Let's say we want to cluster based on multiple columns
columns_to_cluster = df.columns  # Add more columns as needed

# Define the range of clusters to test
cluster_range = range(1, 11)
sse_list = []
col1 = 'Flavanoids'
col2 = 'OD280'
# Extract data from the specified columns
data = df[[col1, col2]].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Calculate SSE for each number of clusters in the range
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(X_scaled)
    sse_list.append(kmeans.inertia_)

# Plot the SSE for each number of clusters
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, sse_list, marker='o')
plt.title(f'Elbow Method for Columns: {col1} and {col2}')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# Clear the SSE list for the next pair of columns
sse_list = []
