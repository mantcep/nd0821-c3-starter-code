# Model Card

## Model Details
The model is a boosting classifier (`xgboost.XGBClassifier`). As part of model training, simple Bayesian hyperparameter tuning is performed using `skopt.BayesSearchCV` implementation. The tuned hyperparameters are: `eta`, `gamma`, `max_depth`, `min_child_weight` and `subsample`.

## Intended Use
This model should be used to predict whether an individual's salary is above or below 50K based off a handful of attributes about the individual.

## Training Data
The data set is based on a data set from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The data set has 32561 rows, and an 80-20 split was used to break this into a train and test set. No stratification was done. The labels were mapped to 1 for ">=50K" and 0 for "<50K".

## Evaluation Data
During hyperparameter tuning, model is evaluated using cross-validation with 5 splits and the evaluation metric is the F1-score.

## Metrics
F1 score, precision and recall were used as metrics to evaluate model performance. The metrics on the test dataset are as follows:
- F1 score – 0.61;
- Precision – 0.69;
- Recall – 0.54.

## Ethical Considerations
The model uses sensitive information, such as an individual's race and gender, therefore the user of the model should verify whether such data can be used.

## Caveats and Recommendations
Currently, all features that are available in the data set are used for modelling, however, the model might benefit from feature selection, as well as additional feature engineering. Moreover, experimenting with additional models (e.g., bagging) and more thorough tuning of hyperparameters should lead to better model performance.
