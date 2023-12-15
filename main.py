from featureEngineering import FeatureEngineering
import pandas as pd
from plotting import generate_plots


def main():
    fileLocation = "C:\\Users\\ozand\\Documents\\Python " \
                   "Projects\\liveOpsGamingUserDataPrediction\\ml_project_2023_cltv_train.xlsx"
    original_data = pd.read_excel(fileLocation, engine='openpyxl')

    # Create a copy of the data for feature engineering
    data_with_features = original_data.copy()

    generate_plots(original_data, include_new_features=False)

    # Weight values are gathered from Correlation to Revenue
    weights = {'banner_cnt': 0.75, 'is_cnt': 0.61, 'rv_cnt': 0.69}

    # Apply feature engineering
    feature_engineer = FeatureEngineering(data_with_features)
    data_with_features = feature_engineer.execute_all(weights)  # Add weights as needed

    # Generate plots for data with new features
    generate_plots(data_with_features, include_new_features=True)


if __name__ == "__main__":
    main()
