import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreProcessing:
    def __init__(self, data):
        self.data = data
        del self.data["country"]
        del self.data["lang"]
        del self.data["_id"]
        print(list(self.data.head()))
        self.original_dtypes = data.dtypes # Storing the original data types to send dataProcessing

    def preprocess_data(self):
        print("Original column types:", self.data.dtypes.value_counts())

        numerical_cols = [col for col in self.data.columns if self.data[col].dtype in ['int64', 'float64', 'bool']]
        categorical_cols = [col for col in self.data.columns if self.data[col].dtype == 'object']

        # Debug: Print counts of numerical and categorical columns
        print("Numerical columns count:", len(numerical_cols))
        print("Categorical columns count:", len(categorical_cols))

        # Numerical data preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Categorical data preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ], remainder='passthrough')

        # Transform the data using the preprocessor and convert it back to a DataFrame
        processed_array = preprocessor.fit_transform(self.data)

        print("Processed array shape:", processed_array.shape)

        onehot_columns = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
        print("One-hot columns:", len(onehot_columns))

        new_column_names = numerical_cols + list(onehot_columns)
        print("Total new columns:", len(new_column_names))

        if len(new_column_names) != processed_array.shape[1]:
            raise ValueError("Mismatch in the number of columns after preprocessing")

        self.data = pd.DataFrame(processed_array, columns=new_column_names, index=self.data.index)
        return self.data
