import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load your data
df = pd.read_csv('wine-clustering.csv')

# Select 3 columns for clustering
columns_to_cluster = ['Malic_Acid', 'Hue', 'OD280']  # Replace with your actual column names
df_selected = df[columns_to_cluster]

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)
labels = kmeans.labels_

# Visualize the clusters in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df_scaled[:, 0], df_scaled[:, 1], df_scaled[:, 2], c=labels, cmap='viridis', marker='o')

# Add labels and title
ax.set_xlabel(columns_to_cluster[0])
ax.set_ylabel(columns_to_cluster[1])
ax.set_zlabel(columns_to_cluster[2])
ax.set_title('KMeans Clustering with 3 Columns')

plt.show()
