"""Shared dense matrix validation for NumPy reduction algorithms."""

import numpy as np


def as_real_matrix(X, name):
    """Convert array-like input to a nonempty, finite, real float64 matrix."""
    try:
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must contain real values, not complex values.")
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real numeric 2D matrix: {exc}") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got {matrix.ndim} dimensions.")
    if 0 in matrix.shape:
        raise ValueError(f"{name} must not be empty (samples and features are required).")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values (no NaN or infinity).")
    return matrix


def as_binary_labels(y, name, n_samples):
    """Validate a one-dimensional 0/1 target without silently recoding labels."""
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError(f"{name} must be a 1D array of length {n_samples}.")
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all():
        raise ValueError(f"{name} must contain finite numeric binary labels 0 and 1.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{name} must contain only binary labels 0 and 1.")
    return labels.astype(np.int64, copy=True)
