"""PCA via sample covariance eigendecomposition, with NumPy as the only dependency.

Fit on training data only. PCA centers features but does not standardize them.
Covariance eigendecomposition costs O(d**2) memory and can lose accuracy on
ill-conditioned inputs; this implementation is intended for the course project.
"""

from numbers import Integral

import numpy as np

from ._validation import as_real_matrix


class ManualPCA:
    """Project real, finite dense matrices onto their leading principal axes.

    Parameters
    ----------
    n_components : int or None, default=None
        Retain 1 through min(n_samples, n_features) axes. None retains that
        maximum, matching sklearn's full-solver convention.

    Attributes
    ----------
    components_ : ndarray of shape (n_components_, n_features_in_)
        Principal axes stored as rows. Signs are fixed by making the largest
        absolute loading in each row nonnegative; signs remain arbitrary
        mathematically, and tied eigenvalues allow different valid bases.
    eigenvalues_ : ndarray of shape (n_features_in_,)
        Full covariance spectrum in descending order, with numerical negatives
        clipped to zero. The remaining variance attributes describe retained PCs.

    Notes
    -----
    Constant training data has zero explained variance. Its variance ratios
    and cumulative ratios are defined here as zeros rather than NaN.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    @staticmethod
    def _as_matrix(X, name):
        """Convert array-like input to a nonempty, finite, real float64 matrix."""
        return as_real_matrix(X, name)

    def _check_fitted(self):
        """Require a successful fit before projection or reconstruction."""
        if not hasattr(self, "components_"):
            raise RuntimeError("ManualPCA is not fitted; call fit(X_train) first.")

    def fit(self, X):
        """Learn the training mean and descending covariance eigenvectors."""
        matrix = self._as_matrix(X, "X")
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must have at least 2 samples for sample covariance.")
        max_components = min(n_samples, n_features)
        k = self.n_components
        if k is None:
            k = max_components
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
            raise ValueError("n_components must be an integer or None.")
        if not 1 <= k <= max_components:
            raise ValueError(
                f"n_components must be between 1 and min(n_samples, n_features) "
                f"= {max_components}; got {k}."
            )
        k = int(k)

        # Work locally: a failed fit cannot replace a previously fitted state.
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean = matrix.mean(axis=0)
                centered = matrix - mean
                covariance = (centered.T @ centered) / (n_samples - 1)
        except FloatingPointError as exc:
            raise ValueError("X is too large for finite covariance; rescale the values.") from exc
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        components = eigenvectors[:, order[:k]].T.copy()

        # Select a deterministic representative of the equivalent +/- axes.
        rows = np.arange(k)
        pivots = np.argmax(np.abs(components), axis=1)
        components *= np.where(components[rows, pivots] < 0, -1.0, 1.0)[:, None]
        total_variance = eigenvalues.sum()
        ratios = (
            eigenvalues[:k] / total_variance
            if total_variance > 0
            else np.zeros(k, dtype=np.float64)
        )

        self.mean_ = mean
        self.components_ = components
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ = eigenvalues[:k].copy()
        self.explained_variance_ratio_ = ratios
        self.cumulative_explained_variance_ = np.cumsum(ratios)
        self.n_components_ = k
        self.n_features_in_ = n_features
        self.n_samples_ = n_samples
        return self

    def transform(self, X):
        """Project samples using the mean and axes learned from training data."""
        self._check_fitted()
        matrix = self._as_matrix(X, "X")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; expected {self.n_features_in_}."
            )
        return (matrix - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        """Fit to X and return its principal component scores."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Reconstruct approximate original features from component scores."""
        self._check_fitted()
        matrix = self._as_matrix(Z, "Z")
        if matrix.shape[1] != self.n_components_:
            raise ValueError(
                f"Z has {matrix.shape[1]} components; expected {self.n_components_}."
            )
        return matrix @ self.components_ + self.mean_

