"""Compare Original/PCA/SVD on a fixed split and save a labelled result table."""

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from cs5487_c_group._core._validation import as_real_matrix
from cs5487_c_group._core.classifier_interface import create_classifiers
from cs5487_c_group._core.config import TABLES_DIR
from cs5487_c_group._core.dimensionality_reduction import create_reducer
from .runner import run_experiment


def compare_reduction_methods(
    X_train, X_test, y_train, y_test, n_components, *, classifier=None,
    classifier_factory=None, output_path=None,
    data_source="unspecified",
    return_results=False,
):
    """Return a three-row DataFrame using the same split and classifier setup.

    A classifier prototype is cloned (sklearn) or deep-copied (custom model),
    leaving it untouched. Alternatively a factory must return fresh, unfitted
    models with the same configuration. Custom prototypes must be unfitted.
    Supplying both options is an error. Default is the temporary baseline.

    k is an integer in 1..min(n_train, n_features), shared by PCA and SVD.
    Metrics use model.predict's threshold; no test-based tuning is performed.
    output_path=None disables CSV writing. The CSV is written only after all
    methods succeed; data_source identifies simulated versus supplied data.
    With return_results=True, return (table, fitted_results_by_method) so later
    error analysis can reuse exactly the models evaluated in the comparison.
    The default return value remains a DataFrame.
    """
    if classifier is not None and classifier_factory is not None:
        raise ValueError("Supply either classifier or classifier_factory, not both.")
    train = as_real_matrix(X_train, "X_train")
    upper = min(train.shape)
    if isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, Integral):
        raise ValueError("n_components must be an integer for the comparison.")
    if not 1 <= n_components <= upper:
        raise ValueError(f"n_components must be between 1 and {upper}.")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a nonempty string.")
    models = create_classifiers(3, classifier=classifier, classifier_factory=classifier_factory)
    rows, fitted_results = [], {}
    for method, model in zip(("Original", "PCA", "SVD"), models):
        result = run_experiment(
            X_train, X_test, y_train, y_test, classifier=model,
            reducer=create_reducer(method, int(n_components)),
        )
        fitted_results[method] = result
        rows.append({
            "method": method, "dimensions": result.dimensions,
            **result.metrics, "data_source": data_source,
        })
    table = pd.DataFrame(rows, columns=[
        "method", "dimensions", "accuracy", "precision", "recall", "f1",
        "pr_auc", "roc_auc", "data_source",
    ])
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False, encoding="utf-8-sig")
    return (table, fitted_results) if return_results else table

