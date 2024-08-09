import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
df = pd.read_csv('wine-clustering.csv')

# Create a histogram for each column
for column in df.columns:
    plt.figure(figsize=(8, 5))

    # Plot the histogram
    sns.histplot(df[column], bins=20, kde=False, color='blue', stat='density', alpha=0.6)

    # Add the KDE line in red
    sns.kdeplot(df[column], color='red', linewidth=2)

    plt.title(f'Distribution of {column} with KDE')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()