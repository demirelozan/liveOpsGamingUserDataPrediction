import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTraining:
    def __init__(self, data):
        self.X_test_reduced = None
        self.X_train_reduced = None
        self.preprocessor = None
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def preprocess_data(self):
        logging.info("Starting data preprocessing.")
        X = self.data.drop('revenue', axis=1)
        y = self.data['revenue']

        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        # Apply one-hot encoding
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ], remainder='passthrough')

        X_processed = self.preprocessor.fit_transform(X)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_processed, y, test_size=0.20,
                                                                                random_state=42)

    def feature_selection_based_on_importance(self, model, threshold=0.01):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Get feature names
            feature_names = self.preprocessor.get_feature_names_out()
            selected_feature_indices = [i for i, imp in enumerate(model.feature_importances_) if imp >= threshold]

            # Filter the data
            self.X_train_reduced = self.X_train[:, selected_feature_indices]
            self.X_test_reduced = self.X_test[:, selected_feature_indices]

            selected_features = [feature_names[i] for i in selected_feature_indices]
            print("Selected features based on importance:", selected_features)
            return self.X_train_reduced, self.X_test_reduced, selected_features
        else:
            print("Feature importance not available for this model type.")
            return self.X_train, self.X_test, []

    def train_linear_regression(self):
        logging.info("Training Linear Regression model.")
        model = LinearRegression()
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_decision_tree(self):
        logging.info("Training Decision Tree model.")
        model = DecisionTreeRegressor(random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_random_forest(self):
        logging.info("Training Random Forest model.")
        model = RandomForestRegressor(random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def train_xgboost(self):
        logging.info("Training XGBoost model.")
        model = XGBRegressor(random_state=42)
        X_train_to_use = self.X_train_reduced if self.X_train_reduced is not None else self.X_train
        model.fit(X_train_to_use, self.y_train)
        return model

    def get_feature_importance(self, model):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = self.preprocessor.get_feature_names_out()
            # Create a Series with feature names and their importance scores
            feature_importances = pd.Series(importances, index=feature_names)
            return feature_importances.sort_values(ascending=False)
        else:
            return "Feature importance not available for this model type."

    def evaluate_model(self, model):
        predictions = model.predict(self.X_test)

        # Regression metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)

        return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R^2": r2}
