import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class DataVisualization:
    def __init__(self, data):
        self.data = data

    # Generate Scatter plots for a given variable against revenue, continuous variables for this case
    def plot_scatter(self, variable, title_prefix, figsize=(10, 6)):
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.data, x=variable, y='revenue', alpha=0.5)
        plt.title(f'{title_prefix} {variable}')
        plt.ylabel('Revenue')
        plt.xlabel(variable)
        plt.show()

    # Generate Bar plots for a given variable against revenue, categorical variables for this case
    def plot_bar(self, variable, title_prefix, figsize=(30, 8)):
        """Generate bar plots for a given categorical variable against revenue."""
        plt.figure(figsize=figsize)
        sns.barplot(data=self.data, x=variable, y='revenue', errorbar=None)
        plt.title(f'{title_prefix} {variable}')
        plt.ylabel('Average Revenue')
        plt.xlabel(variable)

        # Applying truncation only for the Device Brands (to make it more readable)
        if variable == 'device_brand':
            truncated_labels = self.truncate_labels(self.data[variable].astype(str).unique(), 10)
            plt.xticks(ticks=range(len(truncated_labels)), labels=truncated_labels, rotation=90, fontsize=6)
        else:
            plt.xticks(rotation=45, fontsize=8)

        plt.show()

    def truncate_labels(self, labels, max_length):
        """Truncate labels to a maximum character length."""
        return [str(label)[:max_length] + '...' if len(str(label)) > max_length else str(label) for label in labels]

    # Generate a Heatmap to show correlations with revenue
    def plot_heatmap(self, figsize=(12, 8)):
        plt.figure(figsize=figsize)
        numeric_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix[['revenue']].sort_values(by='revenue', ascending=False),
                    annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation of Features with Revenue')
        plt.show()

    def perform_clustering(self, features, n_clusters=3):
        """Perform KMeans clustering and add cluster labels to the data."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.data[features])
        return kmeans.labels_
