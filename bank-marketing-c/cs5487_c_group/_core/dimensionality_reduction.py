"""Construct the feature representations used by comparison experiments."""

from .pca_manual import ManualPCA
from .svd_manual import ManualSVDReducer


def create_reducer(method, n_components=None):
    """Return a fresh PCA/SVD reducer, or None for Original features."""
    if method == "Original":
        return None
    if method == "PCA":
        return ManualPCA(n_components)
    if method == "SVD":
        return ManualSVDReducer(n_components)
    raise ValueError("method must be Original, PCA or SVD.")

