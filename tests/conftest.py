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
def model_trained_on_sample_data(data_sample, cat_features):
    """Fixture for model trained on data sample."""
    X = data_sample
    y = data_sample.pop('salary')

    return train_model(X, y, cat_features)
