"""Script to train machine learning model."""

import pandas as pd
from joblib import dump
from loguru import logger
from sklearn.model_selection import train_test_split

from ml.model import (
    train_model,
    compute_model_metrics,
    compute_metrics_on_cat_slices
)

pd.set_option('display.max_rows', None)

X = pd.read_csv('data/census.csv')
y = X.pop('salary').map({'<=50K': 0, '>50K': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Train and save a model.
model = train_model(X_train, y_train, cat_features)
y_test_pred = model.predict(X_test)

test_metrics = compute_model_metrics(y_test, y_test_pred)
logger.info(
    f"Test precision, recall, f1-score: {test_metrics}."
)

test_metrics_on_slices = compute_metrics_on_cat_slices(
    X_test,
    y_test,
    y_test_pred,
    cat_features
)
logger.info(f"Test performance on slices:\n{test_metrics_on_slices}")

dump(model, 'model/model.joblib')
