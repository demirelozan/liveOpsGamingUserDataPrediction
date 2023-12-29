import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler


class DataProcessing:
    def __init__(self, data):
        self.data = data
        print("The type of self.data before entering cap_outiers() is: " + str(type(self.data)))

    def cap_outliers(self, columns, lower_quantile=0.01, upper_quantile=0.99):
        print("The type of self.data after entering cap_outliers() is: " + str(type(self.data)))
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

        for column in columns:
            if column not in self.data.columns:
                raise ValueError(f"Column {column} not found in data")

            # Ensure the column is numeric
            if not np.issubdtype(self.data[column].dtype, np.number):
                raise TypeError(f"Column {column} must be numeric to cap outliers")

            lower_limit = self.data[column].quantile(lower_quantile)
            upper_limit = self.data[column].quantile(upper_quantile)
            self.data[column] = np.where(self.data[column] < lower_limit, lower_limit, self.data[column])
            self.data[column] = np.where(self.data[column] > upper_limit, upper_limit, self.data[column])

    def apply_robust_scaling(self, columns):
        scaler = RobustScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def winsorize_data(self, columns, limits=(0.01, 0.01)):
        for column in columns:
            self.data[column] = winsorize(self.data[column], limits=limits)

    def apply_log_transformation(self, columns):
        for column in columns:
            self.data[column] = self.data[column].map(lambda x: np.log(x + 1))

    def handle_nans(self):
        # Impute numeric columns with the mean
        num_imputer = SimpleImputer(strategy='mean')
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols] = num_imputer.fit_transform(self.data[numeric_cols])

        # Impute categorical columns with the most frequent value
        cat_imputer = SimpleImputer(strategy='most_frequent')
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.data[categorical_cols] = cat_imputer.fit_transform(self.data[categorical_cols])

    def execute_all(self):
        features_with_outliers = ['max_lvl_no', 'rv_cnt', 'session_cnt', 'banner_cnt', 'gameplay_duration', 'is_cnt']
        self.cap_outliers(features_with_outliers)
        self.apply_robust_scaling(features_with_outliers)
        self.winsorize_data(features_with_outliers)

        return self.data
