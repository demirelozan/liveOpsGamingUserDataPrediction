import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


fileLocation = "C:\\Users\\ozand\\Documents\\Python Projects\\liveOpsGamingUserDataPrediction\\ml_project_2023_cltv_train.xlsx"

# Load the data
data = pd.read_excel(fileLocation, engine='openpyxl')

# Continuous Variables are used create scatter plots against revenue.
# Categorical Variables are used to create bar plots to see how they compare with revenue.
continuous_vars = ['session_cnt', 'gameplay_duration', 'max_lvl_no', 'banner_cnt', 'is_cnt', 'rv_cnt']
categorical_vars = ['os', 'country', 'device_brand']  # Example categorical variables

# Scatter plots for continuous variables
for var in continuous_vars:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=var, y='revenue', alpha=0.5)
    plt.title(f'Revenue vs {var}')
    plt.ylabel('Revenue')
    plt.xlabel(var)
    plt.show()


# Function to truncate labels
def truncate_labels(labels, max_length):
    """Truncate labels to a maximum character length."""
    return [str(label)[:max_length] + '...' if len(str(label)) > max_length else str(label) for label in labels]


# Bar plots for categorical variables
for var in categorical_vars:
    plt.figure(figsize=(30, 6))
    sns.barplot(data=data, x=var, y='revenue', errorbar=None)
    plt.title(f'Average Revenue by {var}')
    plt.ylabel('Average Revenue')
    plt.xlabel(var)

    # Applying truncation only for the Device Brands (to make it more readable), some brand names are too long.
    if var == 'device_brand':
        truncated_labels = truncate_labels(data[var].astype(str).unique(), 10)
        plt.xticks(ticks=range(len(truncated_labels)), labels=truncated_labels, rotation=90, fontsize=6)
    else:
        plt.xticks(rotation=45, fontsize=8)

    plt.show()

# Heatmap to show correlations of all variables with revenue
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

sns.heatmap(correlation_matrix[['revenue']].sort_values(by='revenue', ascending=False),
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlation of Features with Revenue')
print(plt.show())


features = data[['session_cnt', 'gameplay_duration']]

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
data['cluster'] = kmeans.labels_


plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='session_cnt', y='gameplay_duration', hue='cluster', palette='viridis')
plt.title('User Segmentation based on Session Count and Gameplay Duration')
plt.show()

# Creating a Scatterplot Matrix to see the relations of other parameters comparing to each other
selected_columns = ['banner_cnt', 'session_cnt', 'session_length', 'max_lvl_no', 'gameplay_duration',
                    'bonus_cnt', 'repeat_cnt', 'gold_cnt', 'is_cnt', 'rv_cnt', 'max_ses_length', 'avg_ses_length']

data_subset = data[selected_columns]

sns.pairplot(data_subset)
plt.show()