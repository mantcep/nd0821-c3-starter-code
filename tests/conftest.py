import pytest
import pandas as pd


@pytest.fixture
def data():
    """Fixture for loading the raw data."""
    df = pd.read_csv('data/census.csv')
    return df
