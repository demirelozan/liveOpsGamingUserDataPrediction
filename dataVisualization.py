import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import pandas as pd


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

    @staticmethod
    def truncate_labels(labels, max_length):
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
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(self.data[features])
        return kmeans.labels_

    def plot_correlation_matrix(self):
        # Encode categorical variables
        encoded_data = self.data.copy()
        for column in encoded_data.select_dtypes(include=['object']).columns:
            encoded_data[column] = LabelEncoder().fit_transform(encoded_data[column])

        # Plot correlation matrix
        plt.figure(figsize=(15, 10))
        sns.heatmap(encoded_data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix for All Variables')
        plt.show()

    def plot_pairplot(self, selected_columns, figsize=(12, 16), aspect=0.74, plot_kws={'s': 10}):
        """Plot pairwise relationships for selected continuous variables with adjusted plot sizes."""
        sns.set(style="ticks", font_scale=0.75)
        g = sns.pairplot(self.data[selected_columns], height=figsize[0] / len(selected_columns), aspect=aspect,
                         plot_kws=plot_kws)
        g.fig.set_size_inches(*figsize)

        # Rotate x-axis labels and adjust layout
        for ax in g.axes.flatten():
            plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

        plt.tight_layout()
        plt.show()
        sns.reset_orig()

    def plot_categorical_relationship(self, category1, category2, figsize=(10, 8)):
        ct = pd.crosstab(self.data[category1], self.data[category2])
        plt.figure(figsize=figsize)
        sns.heatmap(ct, annot=True, fmt='d', cmap='viridis')
        plt.title(f'Heatmap of {category1} vs {category2}')
        plt.xlabel(category2)
        plt.ylabel(category1)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.show()

    def plot_new_feature_relationships(self, feature_list):
        """Plot relationships of new features with revenue."""
        for feature in feature_list:
            plt.figure(figsize=(10, 6))
            if self.data[feature].dtype == 'float64' or self.data[feature].dtype == 'int64':
                sns.scatterplot(x=self.data[feature], y=self.data['revenue'])
                plt.title(f'Revenue vs {feature}')
            else:
                sns.barplot(x=self.data[feature], y=self.data['revenue'])
                plt.title(f'Average Revenue by {feature}')
            plt.ylabel('Revenue')
            plt.xlabel(feature)
            plt.xticks(rotation=45)
            plt.show()
