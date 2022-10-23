import pandas as pd
from loguru import logger
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer


class tqdm_skopt(object):
    """ Wrapper for progress bar for BayesSearchCV

    https://github.com/scikit-optimize/scikit-optimize/issues/674
    """
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def train_model(X_train, y_train, cat_features, n_iter=50):
    """Train a machine learning model."""
    cat_column_selector = make_column_selector(f"^{'$|^'.join(cat_features)}$")
    model = make_pipeline(
        ColumnTransformer([(
            "cat_features_ohe",
            OneHotEncoder(handle_unknown='ignore'),
            cat_column_selector
        )]),
        XGBClassifier(random_state=42, nthread=1)
    )
    HYPERPARAMETER_RANGES = {
        'xgbclassifier__eta': Real(0, 1),
        'xgbclassifier__gamma': Real(0.000001, 100, 'log-uniform'),
        'xgbclassifier__max_depth': Integer(2, 10),
        'xgbclassifier__min_child_weight': Real(0.000001, 100, 'log-uniform'),
        'xgbclassifier__subsample': Real(0.25, 1)
    }
    opt = BayesSearchCV(
        model,
        HYPERPARAMETER_RANGES,
        n_iter=n_iter,
        random_state=42,
        scoring='f1',
        cv=5,
        n_jobs=-1,
    )
    opt.fit(X_train, y_train,
            callback=[tqdm_skopt(total=50, desc="Performing HPO")])
    logger.info(f"Best f1 score: {opt.best_score_}")
    logger.info(f"Best hyperparameters: {opt.best_params_}")
    return opt.best_estimator_


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
    return pd.DataFrame(results).round(2)
