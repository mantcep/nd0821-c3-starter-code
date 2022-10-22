import pytest
import pandas as pd

from ml.model import train_model


@pytest.fixture
def data():
    """Fixture for loading the raw data."""
    df = pd.read_csv('data/census.csv')
    return df


@pytest.fixture
def data_sample(data):
    """Fixture for small sample of the data."""
    return data.head(20)


@pytest.fixture
def X_sample(data_sample):
    """Fixture for small sample of X."""
    return data_sample.drop(columns='salary')


@pytest.fixture
def y_sample(data_sample):
    """Fixture for small sample of y."""
    return data_sample['salary'].map({'<=50K': 0, '>50K': 1})


@pytest.fixture
def cat_features():
    """Fixture for categorical features."""
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def model_trained_on_sample_data(X_sample, y_sample, cat_features):
    """Fixture for model trained on data sample."""

    return train_model(X_sample, y_sample, cat_features)
