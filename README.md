# Wine Clustering with K-Means and PCA

This project demonstrates how to use Principal Component Analysis (PCA) and K-Means clustering to categorize different types of wine based on their chemical properties.

## Overview

- **PCA**: Applied to reduce the dimensionality of the wine dataset, highlighting key features.
- **K-Means Clustering**: Used to group the wines based on their chemical properties, helping uncover patterns and similarities among different wine types.

## Steps

### 1. Choosing the Optimal Number of Components
To determine the optimal number of components, we analyzed the explained variance ratio:

![optimal_component](https://github.com/user-attachments/assets/f325d69f-189f-49da-9dd8-42d54621a247)

Based on the analysis, we chose 2 components.

### 2. Applying K-Means on PCA Components
We applied K-Means clustering on the selected PCA components:

![kmeans_pca](https://github.com/user-attachments/assets/a07f6ba6-6966-47fc-8878-f849708c1137)

### 3. Comparison with Labeled Data
To evaluate the clustering results, we compared them with the actual wine labels:

![pca_compare](https://github.com/user-attachments/assets/f05e53ce-4d5a-4ac9-9b3d-1559c75513a4)

### 4. Clustering Results
The results were assessed using a confusion matrix:

![confusion1](https://github.com/user-attachments/assets/9e8ce04a-399a-4044-8599-773025945614)

### 5. PCA Loading
Finally, we examined the PCA loading to understand the contribution of each feature:

![pca_loading](https://github.com/user-attachments/assets/fdfee60b-fb09-4278-8146-f25de27320a2)

## How to Run the Code

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook wine_clustering.ipynb
    ```

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

## Conclusion

This project illustrates how PCA and K-Means clustering can be effectively used to group and analyze wines based on their chemical properties, offering insights into the underlying structure of the data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

