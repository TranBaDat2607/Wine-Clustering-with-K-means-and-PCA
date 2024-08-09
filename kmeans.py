import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('wine-clustering.csv')

# Let's say we want to cluster based on multiple columns
columns_to_cluster = df.columns  # Add more columns as needed

for col1 in range(len(columns_to_cluster)):
    for col2 in range(col1 + 1, len(columns_to_cluster)):

        # Extract data from the specified columns
        data = df[[columns_to_cluster[col1], columns_to_cluster[col2]]].values

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)

        # Number of clusters
        k = 3

        # Initialize KMeans with K-means++ initialization
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)

        # Track SSE across iterations
        previous_sse = None
        current_sse = None

        iteration = 0
        # Iterate until convergence (stop when the SSE stops decreasing significantly)
        while previous_sse is None or (previous_sse - current_sse) > 1e-4:
            # Fit the KMeans model
            kmeans.fit(X_scaled)

            # Calculate SSE for the current iteration
            previous_sse = current_sse
            current_sse = kmeans.inertia_

            iteration += 1
            # Print SSE for the current iteration
            print(
                f"Columns: {columns_to_cluster[col1]}, {columns_to_cluster[col2]} - Iteration {iteration} - SSE: {current_sse}")

        # Final SSE after convergence
        print(f"Columns: {columns_to_cluster[col1]}, {columns_to_cluster[col2]} - Final SSE: {current_sse}")
