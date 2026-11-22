"""Shared training-only experiment execution independent of data loading."""

from dataclasses import dataclass

import numpy as np

from cs5487_c_group._core._validation import as_binary_labels, as_real_matrix
from cs5487_c_group._core.classifier_interface import create_classifier
from cs5487_c_group._core.evaluation import evaluate_classifier


@dataclass
class ExperimentResult:
    """Metrics and fitted objects retained for later plots/error analysis."""

    model: object
    reducer: object
    metrics: dict
    dimensions: int


def run_experiment(X_train, X_test, y_train, y_test, classifier=None, reducer=None):
    """Fit a representation and classifier on training data, evaluate on test.

    Supplied classifier/reducer instances are fitted in place. Use fresh
    instances per experiment. Input arrays are copied to protect the caller
    from external estimators that modify their inputs. No splitting or scaling
    is performed here; these belong to the data interface/provider.
    """
    train = as_real_matrix(X_train, "X_train").copy()
    test = as_real_matrix(X_test, "X_test").copy()
    if train.shape[1] != test.shape[1]:
        raise ValueError("X_train and X_test must have the same feature dimensions.")
    train_y = as_binary_labels(y_train, "y_train", len(train))
    test_y = as_binary_labels(y_test, "y_test", len(test))
    if np.unique(train_y).size != 2 or np.unique(test_y).size != 2:
        raise ValueError("Training and test labels must both contain classes 0 and 1.")
    if reducer is not None:
        if not all(callable(getattr(reducer, name, None)) for name in ("fit_transform", "transform")):
            raise ValueError("Reducer must implement fit_transform and transform.")
        reduced_train = as_real_matrix(reducer.fit_transform(train), "reduced X_train")
        reduced_test = as_real_matrix(reducer.transform(test), "reduced X_test")
        if len(reduced_train) != len(train) or len(reduced_test) != len(test):
            raise ValueError("Reducer must preserve the number and order of samples.")
        if reduced_train.shape[1] != reduced_test.shape[1]:
            raise ValueError("Reduced train/test feature dimensions must match.")
        train, test = reduced_train, reduced_test
    model = create_classifier() if classifier is None else classifier
    if not callable(getattr(model, "fit", None)):
        raise ValueError("Classifier must implement fit.")
    model.fit(train, train_y)
    return ExperimentResult(
        model=model, reducer=reducer,
        metrics=evaluate_classifier(model, test, test_y), dimensions=train.shape[1],
    )
