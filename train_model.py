"""Script to train machine learning model."""

import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split


def process_data(
    train: pd.DataFrame,
    categorical_features: List[str],
    label: str,
    training: bool
) -> tuple:
    # Process the test data with the process_data function.
    # TODO: Define output properly
    pass


# Add the necessary imports for the starter code.

# Add code to load in the data.
data = None

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
