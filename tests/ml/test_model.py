from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ml.model import compute_model_metrics, inference


def test_trained_model_is_pipeline(model_trained_on_sample_data):
    """Test that trained model is a pipeline."""
    assert isinstance(model_trained_on_sample_data, Pipeline)


def test_trained_model_is_fitted(model_trained_on_sample_data):
    """Test that the trained model is fitted."""
    check_is_fitted(model_trained_on_sample_data)


def test_compute_model_metrics():
    """Test computation of model metrics."""
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 0, 1, 1]
    expected_precision = 0.6
    expected_recall = 0.6
    expected_fbeta = 0.6

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == expected_precision
    assert recall == expected_recall
    assert fbeta == expected_fbeta


def test_inference_expected_output_rows(
    model_trained_on_sample_data,
    X_sample
):
    """Test that inference function returns expected number of rows."""
    preds = inference(model_trained_on_sample_data, X_sample)

    assert preds.shape[0] == X_sample.shape[0]
