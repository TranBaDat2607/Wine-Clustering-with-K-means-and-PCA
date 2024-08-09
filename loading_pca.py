import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your data
df_unlabeled = pd.read_csv('wine-clustering.csv')

# Scale the data
scaler = StandardScaler()
df_unlabeled_scaled = scaler.fit_transform(df_unlabeled)

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
df_unlabeled_pca = pca.fit_transform(df_unlabeled_scaled)

# Get the PCA components (loadings)
pca_components = pca.components_

# Create a DataFrame for better visualization and analysis
pca_loadings_df = pd.DataFrame(pca_components, columns=df_unlabeled.columns, index=[f'PC{i+1}' for i in range(pca.n_components)])

# Display the loadings
print(pca_loadings_df)
print(sum(pca_loadings_df[0]))
# Heatmap of the loadings
plt.figure(figsize=(12, 6))
sns.heatmap(pca_loadings_df, cmap="coolwarm", annot=True, center=0)
plt.title('PCA Loadings')
plt.show()
