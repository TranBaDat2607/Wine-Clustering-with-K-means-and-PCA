import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the data
df_unlabeled = pd.read_csv('wine-clustering.csv')
df_labeled = pd.read_csv('data_with_label.csv')

# Select the two columns to be clustered
col1 = 'Total_Phenols'
col2 = 'Flavanoids'
X_unlabeled = df_unlabeled[[col1, col2]]
X_labeled = df_labeled[[col1, col2]]
true_labels = df_labeled['Label']

# Standardize the data
scaler = StandardScaler()
X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
X_labeled_scaled = scaler.fit_transform(X_labeled)

# Apply K-Means clustering with K-Means++ initialization
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)  # Adjust the number of clusters as needed
kmeans.fit(X_unlabeled_scaled)
clusters_unlabeled = kmeans.predict(X_unlabeled_scaled)
clusters_labeled = kmeans.predict(X_labeled_scaled)

# Add the cluster labels to the DataFrame
df_unlabeled['Cluster'] = clusters_unlabeled

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, clusters_labeled)

# Use the Hungarian algorithm to find the best assignment
row_ind, col_ind = linear_sum_assignment(-conf_matrix)
optimal_assignment = col_ind

# Map the clusters to the true labels based on the optimal assignment
mapped_clusters = np.zeros_like(clusters_labeled)
for i, cluster in enumerate(optimal_assignment):
    mapped_clusters[clusters_labeled == cluster] = i

# Compute the confusion matrix with mapped clusters
mapped_conf_matrix = confusion_matrix(true_labels, mapped_clusters)

# Calculate the number of correctly and incorrectly predicted data points
correct_predictions = np.sum(mapped_clusters == true_labels)
incorrect_predictions = np.sum(mapped_clusters != true_labels)

# Print the results
print(f"Correctly predicted data points: {correct_predictions}")
print(f"Incorrectly predicted data points: {incorrect_predictions}")

# Calculate clustering accuracy using ARI and NMI
ari_score = adjusted_rand_score(true_labels, clusters_labeled)
nmi_score = normalized_mutual_info_score(true_labels, clusters_labeled)

# Print the clustering evaluation scores
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

# Print the final SSE
print(f"Final SSE: {kmeans.inertia_}")

# Visualize the actual data with true labels
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_labeled[col1], X_labeled[col2], c=true_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.title('Actual Data with True Labels')
plt.xlabel(col1)
plt.ylabel(col2)

# Visualize the clustered data with predicted labels
plt.subplot(1, 2, 2)
plt.scatter(X_labeled[col1], X_labeled[col2], c=mapped_clusters, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            s=200, c='red', marker='x')  # Centroids
plt.title('K-Means Clustering with Predicted Labels')
plt.xlabel(col1)
plt.ylabel(col2)

plt.show()

# Plot the confusion matrix using seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(mapped_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Remove the top and right spines
sns.despine(top=True, right=True)

plt.show()
print(mapped_conf_matrix)
