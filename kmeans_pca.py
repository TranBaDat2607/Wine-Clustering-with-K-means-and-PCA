import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import numpy as np
from scipy.optimize import linear_sum_assignment

# Load your data
df_unlabeled = pd.read_csv('wine-clustering.csv')
df_labeled = pd.read_csv('data_with_label.csv')

# Assuming your data needs standardization
scaler = StandardScaler()
df_unlabeled_scaled = scaler.fit_transform(df_unlabeled)
df_labeled_scaled = scaler.fit_transform(df_labeled.drop('Label', axis=1))

# Apply PCA to both datasets
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
df_unlabeled_pca = pca.fit_transform(df_unlabeled_scaled)
df_labeled_pca = pca.fit_transform(df_labeled_scaled)

# Determine the optimal number of clusters (you might have done this in your code)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters_unlabeled = kmeans.fit_predict(df_unlabeled_pca)
kmeans_labeled = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters_labeled = kmeans_labeled.fit_predict(df_labeled_pca)

# Get the true labels from the labeled dataset
true_labels = df_labeled['Label'].values

# Calculate clustering accuracy using ARI and NMI
ari_score = adjusted_rand_score(true_labels, clusters_labeled)
nmi_score = normalized_mutual_info_score(true_labels, clusters_labeled)

# Print the clustering evaluation scores
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, clusters_labeled)

# Use the Hungarian algorithm to find the best assignment
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
optimal_assignment = col_ind

# Map the clusters to the true labels based on the optimal assignment
mapped_clusters = np.zeros_like(clusters_labeled)
for i, cluster in enumerate(optimal_assignment):
    mapped_clusters[clusters_labeled == cluster] = i

# Calculate the number of correctly and incorrectly predicted data points
correct_predictions = np.sum(mapped_clusters == true_labels)
incorrect_predictions = np.sum(mapped_clusters != true_labels)

# Print the results
print(f"Correctly predicted data points: {correct_predictions}")
print(f"Incorrectly predicted data points: {incorrect_predictions}")

# Visualize the clustered data with labels
plt.figure(figsize=(8, 6))
plt.scatter(df_labeled_pca[:, 0], df_labeled_pca[:, 1], c=true_labels, cmap='viridis', marker='o', alpha=0.6, label='True Labels')
plt.scatter(df_labeled_pca[:, 0], df_labeled_pca[:, 1], c=mapped_clusters, cmap='jet', marker='x', alpha=0.6, label='Mapped Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering with PCA (Labeled Data)')
plt.legend()
plt.show()

# Count the number of data points in each cluster for the labeled data
cluster_counts_labeled = np.bincount(clusters_labeled)

# Print the number of data points in each cluster
for i, count in enumerate(cluster_counts_labeled):
    print(f"Cluster {i}: {count} data points")

plt.figure(figsize=(8, 6))
plt.scatter(df_unlabeled_pca[:, 0], df_unlabeled_pca[:, 1], c=clusters_unlabeled, cmap='jet', marker='o', alpha=0.6, label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='o', s=200, label='Cluster Centers')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KMeans Clustering with PCA (Unlabeled Data)')
plt.legend()
plt.show()

print(kmeans.inertia_)