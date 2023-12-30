import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelTraining:
    def __init__(self, data, target_column='revenue'):
        self.X_test_reduced = None
        self.X_train_reduced = None
        self.preprocessor = None
        self.data = data
        self.target_column = target_column
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def separate_features_and_target(self):
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        return X, y

    def apply_one_hot_encoding(self, X, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ], remainder='passthrough')

        return self.preprocessor.fit_transform(X)

    def split_data(self, test_size=0.20, random_state=42):
        X, y = self.separate_features_and_target()
        X_processed = self.apply_one_hot_encoding(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state)

    def perform_cross_validation(self, model, cv_folds=5):
        mse_scores = cross_val_score(model, self.X_train, self.y_train, scoring='neg_mean_squared_error', cv=cv_folds)
        r2_scores = cross_val_score(model, self.X_train, self.y_train, scoring='r2', cv=cv_folds)

        rmse_scores = np.sqrt(-mse_scores)
        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)

        return {'Average RMSE': avg_rmse, 'Average R^2': avg_r2}

    def feature_selection_based_on_importance(self, model, threshold=0.01):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

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

    def train_gradient_boosting(self):
        logging.info("Training Gradient Boost Model.")
        model = GradientBoostingRegressor(random_state=42)
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

        # Performance Evaluation Metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, predictions)

        return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R^2": r2}
