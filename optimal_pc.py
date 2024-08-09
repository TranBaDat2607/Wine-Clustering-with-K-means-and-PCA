import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load your data
df = pd.read_csv('wine-clustering.csv')

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA()
pca.fit(df_scaled)

# Print all eigenvalues
print("Eigenvalues:")
print(pca.explained_variance_)
print(sum(pca.explained_variance_[:2])/sum(pca.explained_variance_))

# Fit PCA on scaled data
pca = PCA().fit(df_scaled)

# Plotting the Cumulative Summation of the Explained Variance
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), explained_variance_ratio_cumsum, label='Cumulative Explained Variance')
plt.scatter(range(1, len(pca.explained_variance_ratio_) + 1), explained_variance_ratio_cumsum, color='red')  # Add red dots
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.legend()
plt.show()
