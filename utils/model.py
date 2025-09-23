# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def data_split(dataset, target_column='is_positive_growth_3m_future'):
    """Splits the dataset into training, validation, and test sets based on existing `split` column."""

    if 'split' not in dataset.columns:
        raise ValueError("The dataset must contain a 'split' column with values 'train', 'validation', and 'test'.")

    # Define features (X) and target (y) per split
    #train_df = dataset[dataset.split.isin(['train'])].copy(deep=True).drop(columns=[target_column, 'split'])
    #valid_df = dataset[dataset.split.isin(['validation'])].copy(deep=True).drop(columns=[target_column, 'split'])
    train_valid_df = dataset[dataset.split.isin(['train','validation'])].copy(deep=True).drop(columns=[target_column, 'split'])
    test_df =  dataset[dataset.split.isin(['test'])].copy(deep=True).drop(columns=[target_column, 'split'])

    #y_train = dataset[dataset.split.isin(['train'])].copy(deep=True)[target_column]
    #y_valid = dataset[dataset.split.isin(['validation'])].copy(deep=True)[target_column]
    y_train_valid = dataset[dataset.split.isin(['train','validation'])].copy(deep=True)[target_column]
    y_test =  dataset[dataset.split.isin(['test'])].copy(deep=True)[target_column]


    # Standardize the features
    scaler = StandardScaler()
    X_train_valid = scaler.fit_transform(train_valid_df)
    X_test = scaler.transform(test_df)

    return X_train_valid, y_train_valid, X_test, y_test, scaler


def get_predictions_correctness(y_test, y_pred, to_predict):

    print(f'Prediction column: {to_predict}')


    is_correct = (y_test == y_pred).astype(int)

    print(is_correct.value_counts())
    print(is_correct.value_counts()/len(y_test))
    print('-'*24)
    print(confusion_matrix(y_test, y_pred))
    print('-'*24)

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return


def fit_decision_tree(X, y, max_depth=5):
    """Fit a simple dt classifier."""

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=max_depth,
                                  random_state=24)

    # Fit the classifier to the training data
    clf.fit(X, y)

    return clf


def fit_random_forest(X, y, max_depth=5):
    """Fit a simple rf classifier."""

    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(max_depth=max_depth,
                                  random_state=24,
                                  n_jobs=-1)

    # Fit the classifier to the training data
    clf.fit(X, y)

    return clf

def fit_knn(X, y, n_neighbors=5):
    """Fit a simple knn classifier."""

    # Initialize the k-Nearest-Neighbors Classifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                  n_jobs=-1)

    # Fit the classifier to the training data
    clf.fit(X, y)

    return clf



def tune_and_select_best_classifier(X_train_valid, y_train_valid, X_test, y_test, scorer=None):
    """Tune and select the best classifier among Decision Tree, Random Forest, and KNN using GridSearchCV.

    This function performs hyperparameter tuning for three classifiers using time series cross-validation.
    The best model is selected based on a user-provided or default precision scorer (positive class).
    Saves model info, metrics, and configuration to a YAML file.

    Parameters
    ----------
    X_train_valid : array-like
        Training features.
    y_train_valid : array-like
        Training labels.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    scorer : callable, optional
        Scoring function (default: precision of positive class).
    yaml_path : str, optional
        File path for exporting model info as YAML.

    Returns
    -------
    best_model : sklearn estimator
        Best trained classifier.
    best_model_name : str
        Name of the best-performing model.
    best_params : dict
        Best hyperparameters found.
    best_metric : float
        Test set value of the selection metric.
    """
    # Set default scorer: precision for positive class (label=1)
    if scorer is None:
        scorer = make_scorer(precision_score, pos_label=1)

    # Define classifiers and grids
    dt = DecisionTreeClassifier(random_state=24)
    rf = RandomForestClassifier(random_state=24)
    knn = KNeighborsClassifier()

    param_grid_dt = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    # Set up GridSearchCV
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=tscv, scoring=scorer, n_jobs=-1)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=tscv, scoring=scorer, n_jobs=-1)
    grid_knn = GridSearchCV(knn, param_grid_knn, cv=tscv, scoring=scorer, n_jobs=-1)

    # Fit models
    grid_dt.fit(X_train_valid, y_train_valid)
    grid_rf.fit(X_train_valid, y_train_valid)
    grid_knn.fit(X_train_valid, y_train_valid)

    models = {
        'DecisionTree': grid_dt,
        'RandomForest': grid_rf,
        'KNN': grid_knn
    }

    # Evaluate on test set using the scorer metric
    results = {}
    for name, grid in models.items():

        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(X_test)

        test_metric = round(scorer._score_func(y_test, y_pred, pos_label=1),3)
        acc = round(accuracy_score(y_test, y_pred),3)

        results[name] = {
            'model': best_estimator,
            'params': grid.best_params_,
            'metric': test_metric,
            'accuracy': acc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

    # Select the best model based on test metric
    best_name = max(results, key=lambda x: results[x]['metric'])
    best_info = results[best_name]
    best_model = best_info['model']

    print(f"{best_name} best params:", best_info['params'])
    print(f"{best_name} test metric (precision):", best_info['metric'])

    # Export YAML info
    yaml_dict = {
        'model_name': best_name,
        'best_params': best_info['params'],
        'precision_score of positive class (test-set)': best_info['metric'],
        'accuracy': best_info['accuracy'],
        'classification_report': best_info['classification_report'],
        'confusion_matrix': best_info['confusion_matrix']
    }

    # Define folder for saving models
    custom_folder = '../saved_models/'
    os.makedirs(custom_folder, exist_ok=True)


    with open(custom_folder + '/best_model_parameters.yaml', 'w') as f:
        yaml.dump(yaml_dict, f)


    # Format today's date as YYYYMMDD
    today_str = datetime.now().strftime('%Y%m%d')


    # Build filename with date
    model_filename = f"model_{best_name.lower()}_trainingDate_{today_str}.joblib"
    model_path = os.path.join(custom_folder, model_filename)

    # Save model
    joblib.dump(best_model, model_path)

    print(f"Model saved to {model_path}")

    return best_info['model'], best_name, best_info['params'], best_info['metric']
