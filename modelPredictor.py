import pandas as pd
from dataPreProcessing import DataPreProcessing
from featureEngineering import FeatureEngineering
import joblib


class ModelPredictor:
    def __init__(self, model_path, test_data_path, selected_features_path):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.selected_features_path = selected_features_path
        self.model = joblib.load(self.model_path)
        self.selected_features = joblib.load(self.selected_features_path)
        self.test_data = pd.read_excel(self.test_data_path)

    def preprocess_and_predict(self):
        data_preprocessor = DataPreProcessing(self.test_data)
        preprocessed_data = data_preprocessor.preprocess_data()

        # Apply feature engineering
        weights = {'banner_cnt': 0.75, 'is_cnt': 0.61, 'rv_cnt': 0.69}
        feature_engineer = FeatureEngineering(preprocessed_data)
        data_with_features = feature_engineer.execute_all(weights)

        # Filter the test data to keep only the selected features
        data_with_selected_features = data_with_features[self.selected_features]

        # Predict
        predictions = self.model.predict(data_with_selected_features)
        return predictions
