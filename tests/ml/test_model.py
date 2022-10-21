from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def test_trained_model_is_pipeline(model_trained_on_sample_data):
    """Test that trained model is a pipeline."""
    assert isinstance(model_trained_on_sample_data, Pipeline)


def test_trained_model_is_fitted(model_trained_on_sample_data):
    """Test that the trained model is fitted."""
    check_is_fitted(model_trained_on_sample_data)
