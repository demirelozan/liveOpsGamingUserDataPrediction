from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ModelTraining:
    def __init__(self, data):
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def split_data(self, test_size=0.20, random_state=42):
        X = self.data.drop('revenue', axis=1)
        y = self.data['revenue']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=random_state)

    def train_model(self):
        # Example: Training a Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate_model(self, model):
        predictions = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
