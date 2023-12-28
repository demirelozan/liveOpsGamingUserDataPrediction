from featureEngineering import FeatureEngineering
from modelTraining import ModelTraining
import pandas as pd
from plotting import generate_plots


def main():
    fileLocation = "C:\\Users\\ozand\\Documents\\Python " \
                   "Projects\\liveOpsGamingUserDataPrediction\\ml_project_2023_cltv_train.xlsx"
    original_data = pd.read_excel(fileLocation, engine='openpyxl')

    # Create a copy of the data for feature engineering
    data_with_features = original_data.copy()

    #    generate_plots(original_data, include_new_features=False)

    # Weight values are gathered from Correlation to Revenue
    weights = {'banner_cnt': 0.75, 'is_cnt': 0.61, 'rv_cnt': 0.69}

    # Apply feature engineering
    feature_engineer = FeatureEngineering(data_with_features)
    data_with_features = feature_engineer.execute_all(weights)  # Add weights as needed

    # Generate plots for data with new features
    #    generate_plots(data_with_features, include_new_features=True)

    print(list(data_with_features.columns))

    model_trainer = ModelTraining(data_with_features)
    model_trainer.preprocess_data()
    #    model_trainer.split_data()

    # Train various models
    lr_model = model_trainer.train_linear_regression()
    dt_model = model_trainer.train_decision_tree()
    rf_model = model_trainer.train_random_forest()
    xgb_model = model_trainer.train_xgboost()
    gbm_model = model_trainer.train_gradient_boosting()

    lr_metrics = model_trainer.evaluate_model(lr_model)

    dt_feature_importance = model_trainer.get_feature_importance(dt_model)
    print("Decision Tree Feature Importance:\n", dt_feature_importance)
    rf_feature_importance = model_trainer.get_feature_importance(rf_model)
    print("Random Forest Feature Importance:\n", rf_feature_importance)
    xgb_feature_importance = model_trainer.get_feature_importance(xgb_model)
    print("Xgboost Feature Importance:\n", xgb_feature_importance)
    gbm_feature_importance = model_trainer.get_feature_importance(gbm_model)
    print("GBM Feature Importance:\n", gbm_feature_importance)

    # Evaluate models
    print("Linear Regression Metrics:", lr_metrics)
    print("Decision Tree Accuracy:", model_trainer.evaluate_model(dt_model))
    print("Random Forest Accuracy:", model_trainer.evaluate_model(rf_model))
    print("XGBoost Accuracy:", model_trainer.evaluate_model(xgb_model))
    print("GBM Accuracy:", model_trainer.evaluate_model(gbm_model))

    # Feature selection based on model importance
    _, _, selected_features = model_trainer.feature_selection_based_on_importance(gbm_model)
    print("Selected Features:", selected_features)

    model_trainer.X_train, model_trainer.X_test, _ = model_trainer.feature_selection_based_on_importance(xgb_model)

    lr_model_selected = model_trainer.train_linear_regression()
    dt_model_selected = model_trainer.train_decision_tree()
    rf_model_selected = model_trainer.train_random_forest()
    xgb_model_selected = model_trainer.train_xgboost()
    gbm_model_selected = model_trainer.train_gradient_boosting()

    # Evaluate the model with selected features
    lr_metrics_selected = model_trainer.evaluate_model(lr_model_selected)
    dt_metrics_selected = model_trainer.evaluate_model(dt_model_selected)
    rf_metrics_selected = model_trainer.evaluate_model(rf_model_selected)
    xgb_metrics_selected = model_trainer.evaluate_model(xgb_model_selected)
    gbm_metrics_selected = model_trainer.evaluate_model(gbm_model_selected)

    print("Linear Regression with Selected Feature Metrics: ", lr_metrics_selected)
    print("Decision Tree with Selected Feature Metrics: ", dt_metrics_selected)
    print("Random Forest with Selected Feature Metrics: ", rf_metrics_selected)
    print("XGBoost with Selected Feature Metrics: ", xgb_metrics_selected)
    print("GBM with Selected Feature Metrics: ", gbm_metrics_selected)


if __name__ == "__main__":
    main()
