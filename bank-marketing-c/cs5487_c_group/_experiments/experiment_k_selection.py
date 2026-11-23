"""PCA component-count sweeps on one fixed training/evaluation split."""

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from cs5487_c_group._core._validation import as_real_matrix
from cs5487_c_group._core.classifier_interface import create_classifiers
from cs5487_c_group._core.config import TABLES_DIR
from .experiment_pca import run_pca_experiment


def _positive_integer(value, name):
    """Validate dimensions/steps without truncating floats or accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def generate_k_values(n_samples, n_features, *, step=2):
    """Generate sorted legal k values including 1 and the maximum dimension.

    Default for upper=12 is [1, 2, 4, 6, 8, 10, 12]; upper=1 yields [1].
    The upper limit is min(n_samples, n_features); PCA needs >=2 samples.
    """
    n_samples = _positive_integer(n_samples, "n_samples")
    n_features = _positive_integer(n_features, "n_features")
    step = _positive_integer(step, "step")
    if n_samples < 2:
        raise ValueError("PCA requires at least 2 training samples.")
    upper = min(n_samples, n_features)
    return sorted({1, upper, *range(step, upper + 1, step)})


def evaluate_pca_over_k(
    X_train, X_test, y_train, y_test, k_values=None, *, classifier=None,
    classifier_factory=None, output_path=None,
    data_source="unspecified",
):
    """Return metrics and retained training variance for each requested k.

    Default k values come from generate_k_values. A custom iterable must be
    nonempty and contain legal integers; duplicates are removed and values
    sorted. All k values are validated before fitting or writing anything.

    Each k uses a fresh PCA and classifier on the same supplied split. The
    cumulative explained variance is computed only from training features.
    This produces a descriptive curve, without choosing a best k. For formal
    model selection supply validation data here, then evaluate the chosen
    configuration on a separate test set.

    Classifier injection follows compare_reduction_methods. output_path=None
    disables saving; data_source labels synthetic versus supplied data.
    """
    if classifier is not None and classifier_factory is not None:
        raise ValueError("Supply either classifier or classifier_factory, not both.")
    train = as_real_matrix(X_train, "X_train")
    default_values = generate_k_values(*train.shape)
    upper = min(train.shape)
    if k_values is None:
        values = default_values
    else:
        if isinstance(k_values, (str, bytes)):
            raise ValueError("k_values must be a nonempty iterable of integers.")
        try:
            requested = list(k_values)
        except TypeError as exc:
            raise ValueError("k_values must be a nonempty iterable of integers.") from exc
        if not requested:
            raise ValueError("k_values must not be empty.")
        values = []
        for k in requested:
            k = _positive_integer(k, "k_values entries")
            if k > upper:
                raise ValueError(f"k_values entries must be between 1 and {upper}; got {k}.")
            values.append(k)
        values = sorted(set(values))
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a nonempty string.")
    models = create_classifiers(len(values), classifier=classifier, classifier_factory=classifier_factory)
    rows = []
    for k, model in zip(values, models):
        result = run_pca_experiment(
            X_train, X_test, y_train, y_test, k, classifier=model,
        )
        rows.append({
            "k": k, **result.metrics,
            "cumulative_explained_variance": float(result.reducer.cumulative_explained_variance_[-1]),
            "data_source": data_source,
        })
    table = pd.DataFrame(rows, columns=[
        "k", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc",
        "cumulative_explained_variance", "data_source",
    ])
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False, encoding="utf-8-sig")
    return table

