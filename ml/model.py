import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def train_model(X_train, y_train, cat_features):
    """Train a machine learning model."""
    cat_column_selector = make_column_selector(f"^{'$|^'.join(cat_features)}$")
    model = make_pipeline(
        ColumnTransformer(
            [("cat_features_ohe", OneHotEncoder(), cat_column_selector)]
        ),
        ExtraTreesClassifier()
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """Compute precision, recall, and F1."""
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions."""
    return model.predict(X)


def compute_metrics_on_cat_slices(X, y, y_preds, cat_features):
    """Compute model metrics on slices of categorical features."""
    results = []
    for cat_feature in cat_features:
        for cat_feature_val in X[cat_feature].unique():
            filter_ = X[cat_feature] == cat_feature_val
            y_slice, y_preds_slice = y[filter_], y_preds[filter_]
            precision, recall, fscore = compute_model_metrics(y_slice,
                                                              y_preds_slice)
            results.append({
                'cat_feature': cat_feature,
                'cat_feature_val': cat_feature_val,
                'precision': precision,
                'recall': recall,
                'fscore': fscore
            })
    return pd.DataFrame(results)
