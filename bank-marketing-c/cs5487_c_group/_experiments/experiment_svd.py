"""Single uncentered SVD experiment; all data is supplied by the caller."""

from cs5487_c_group._core.dimensionality_reduction import create_reducer
from .runner import run_experiment


def run_svd_experiment(X_train, X_test, y_train, y_test, n_components, classifier=None):
    """Fit a fresh SVD on training features, then fit/evaluate the classifier."""
    return run_experiment(
        X_train, X_test, y_train, y_test, classifier=classifier,
        reducer=create_reducer("SVD", n_components),
    )

