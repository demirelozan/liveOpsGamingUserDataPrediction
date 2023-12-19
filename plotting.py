from dataVisualization import DataVisualization


def generate_plots(data, include_new_features=False):
    visualizer = DataVisualization(data)

    if include_new_features:
        # List of new features to plot against revenue
        new_features = ['LevelsPerSession', 'InteractivityPerSession', 'AvgGameplayDurationPerSession',
                        'WeightedAdInteraction', 'AdInteractionPerSession', 'PositiveGameplay', 'Penalty', 'GameEfficiencyRate', 'PenaltyInteractivity']

        # Visualize relationships of new features with revenue
        visualizer.plot_new_feature_relationships(new_features)

        # Adding Revenue to New Features to add to the following plots.
        new_features.append('revenue')
        # Generate correlation matrix for new features
        visualizer.plot_selected_features_correlation(new_features)

        # Generate Heatmap for new features
        visualizer.plot_selected_features_heatmap(new_features)

    else:

        # Continuous Variables are used to create scatter plots against revenue.
        # Categorical Variables are used to create bar plots to see how they compare with revenue.
        continuous_vars = ['session_cnt', 'gameplay_duration', 'max_lvl_no', 'banner_cnt', 'is_cnt', 'rv_cnt']
        categorical_vars = ['os', 'country', 'device_brand']

        for var in continuous_vars:
            visualizer.plot_scatter(var, 'Revenue vs')

        for var in categorical_vars:
            visualizer.plot_bar(var, 'Average Revenue by')

        # Plot heatmap of correlations
        visualizer.plot_heatmap()

        cluster_labels = visualizer.perform_clustering(['session_cnt', 'gameplay_duration'])
        data['cluster'] = cluster_labels
        visualizer.plot_scatter('session_cnt', 'User Segmentation based on Session Count and Gameplay Duration',
                                figsize=(10, 6))

        # Plot the correlation matrix
        visualizer.plot_correlation_matrix()

        # Plot pairwise relationships for selected variables
    #    visualizer.plot_pairplot(['session_cnt', 'gameplay_duration', 'max_lvl_no', 'banner_cnt'], figsize=(15, 15))

    # Plot a relationship between two categorical variables
    #    visualizer.plot_categorical_relationship('os', 'device_brand')
