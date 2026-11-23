"""Extract and describe binary classification mistakes without fitting models.

Supply numeric features in the original feature space, aligned positionally
with y_true/y_pred. For a PCA/SVD model, predict on reduced features first,
then pass the corresponding original feature rows here for interpretation.
No data loading, prediction, retraining, file writing or threshold tuning is
performed by these functions.
"""

import numpy as np
import pandas as pd

from ._validation import as_binary_labels, as_real_matrix

_METADATA_COLUMNS = ("row_position", "sample_id", "y_true", "y_pred", "error_type")


def get_misclassified_samples(X, y_true, y_pred, feature_names=None, *, sample_ids=None):
    """Return FP/FN rows as a DataFrame, preserving input feature order.

    FP means true=0/predicted=1; FN means true=1/predicted=0. Output metadata
    precedes numeric feature columns. row_position is the zero-based position
    in the supplied X. sample_id uses explicit IDs, a DataFrame's index, or
    row_position. Repeated IDs are allowed; row_position remains unambiguous.
    Labels are matched by position, not by pandas index. A no-error result is
    an empty DataFrame with the same schema. X itself must be nonempty.

    DataFrame names are used by default; unnamed arrays get feature_1, etc.
    Names must be unique nonempty strings and cannot collide with metadata.
    """
    matrix = as_real_matrix(X, "X")
    n_samples, n_features = matrix.shape
    truth = as_binary_labels(y_true, "y_true", n_samples)
    predicted = as_binary_labels(y_pred, "y_pred", n_samples)
    if feature_names is None:
        names = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i + 1}" for i in range(n_features)]
    else:
        if isinstance(feature_names, (str, bytes)):
            raise ValueError("feature_names must be a sequence matching X columns.")
        try:
            names = list(feature_names)
        except TypeError as exc:
            raise ValueError("feature_names must be a sequence matching X columns.") from exc
    if len(names) != n_features or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must match X columns with nonempty string names.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    if set(names).intersection(_METADATA_COLUMNS):
        raise ValueError(f"feature_names must not use reserved metadata names: {_METADATA_COLUMNS}.")
    if isinstance(X, pd.DataFrame) and list(X.columns) != names:
        raise ValueError("DataFrame columns must match feature_names in order.")
    if sample_ids is None:
        ids = X.index.to_numpy(copy=True) if isinstance(X, pd.DataFrame) else np.arange(n_samples)
    else:
        ids = np.asarray(sample_ids)
    if ids.ndim != 1 or len(ids) != n_samples or pd.isna(ids).any():
        raise ValueError(f"sample_ids must be a 1D array of length {n_samples} without missing IDs.")
    errors = truth != predicted
    details = pd.DataFrame({
        "row_position": np.arange(n_samples)[errors],
        "sample_id": ids[errors], "y_true": truth[errors], "y_pred": predicted[errors],
        "error_type": np.where(truth[errors] == 0, "FP", "FN"),
    })
    features = pd.DataFrame(matrix[errors], columns=names)
    return pd.concat([details, features], axis=1)


def summarize_misclassifications(X, y_true, y_pred, feature_names=None, *, sample_ids=None):
    """Return {'counts': dict, 'feature_summary': DataFrame} for the same rows.

    counts includes TN/FP/FN/TP and error_rate. false_positive_rate is FP/(TN+FP),
    false_negative_rate is FN/(TP+FN); a rate with no applicable true class is
    None. Both rates can be computed for single-class input when their actual
    denominator is present, independently of AUC evaluation requirements.

    feature_summary contains both error types crossed with each feature:
    error_type, feature, count, mean, median and population std (ddof=0).
    An absent error type has count=0 and NaN statistics, rather than invented
    zero feature values. Statistics use supplied units (possibly standardized)
    and are descriptive; no feature ranking or causal conclusions are made.
    """
    details = get_misclassified_samples(
        X, y_true, y_pred, feature_names, sample_ids=sample_ids,
    )
    # Labels have already been validated together with X by extraction.
    truth, predicted = np.asarray(y_true), np.asarray(y_pred)
    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    counts = {
        "n_samples": len(truth), "n_correct": tn + tp, "n_errors": fp + fn,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "error_rate": (fp + fn) / len(truth),
        "false_positive_rate": fp / (tn + fp) if tn + fp else None,
        "false_negative_rate": fn / (tp + fn) if tp + fn else None,
    }
    rows = []
    for error_type in ("FP", "FN"):
        group = details.loc[details.error_type == error_type]
        for feature in details.columns[len(_METADATA_COLUMNS):]:
            values = group[feature].to_numpy()
            rows.append({
                "error_type": error_type, "feature": feature, "count": len(values),
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=0)) if len(values) else np.nan,
            })
    return {"counts": counts, "feature_summary": pd.DataFrame(rows)}

