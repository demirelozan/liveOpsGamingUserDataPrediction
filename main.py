import pandas as pd
from dataVisualization import DataVisualization


def main():
    fileLocation = "C:\\Users\\ozand\\Documents\\Python " \
                   "Projects\\liveOpsGamingUserDataPrediction\\ml_project_2023_cltv_train.xlsx"
    data = pd.read_excel(fileLocation, engine='openpyxl')

    # Continuous Variables are used to create scatter plots against revenue.
    # Categorical Variables are used to create bar plots to see how they compare with revenue.
    continuous_vars = ['session_cnt', 'gameplay_duration', 'max_lvl_no', 'banner_cnt', 'is_cnt', 'rv_cnt']
    categorical_vars = ['os', 'country', 'device_brand']

    print(data.columns)
    visualizer = DataVisualization(data)

    for var in continuous_vars:
        visualizer.plot_scatter(var, 'Revenue vs')

    for var in categorical_vars:
        visualizer.plot_bar(var, 'Average Revenue by')

    # Plot heatmap of correlations
    visualizer.plot_heatmap()

    # Perform clustering and plot results
    cluster_labels = visualizer.perform_clustering(['session_cnt', 'gameplay_duration'])
    data['cluster'] = cluster_labels
    visualizer.plot_scatter('session_cnt', 'User Segmentation based on Session Count and Gameplay Duration',
                            figsize=(10, 6))


if __name__ == "__main__":
    main()
