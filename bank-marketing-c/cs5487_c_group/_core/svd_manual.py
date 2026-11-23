"""Dense, uncentered SVD reduction using NumPy rather than sklearn internally.

For centered X = U S V.T, PCA covariance eigenvalues equal S**2 / (n-1).
This reducer does not center X, matching TruncatedSVD's feature representation.
It computes an exact economy SVD then retains k axes, not a sparse/randomized
solver; it is intended for course experiments on moderate dense matrices.
"""

from numbers import Integral

import numpy as np

from ._validation import as_real_matrix


class ManualSVDReducer:
    """Reduce dense X to X @ V_k without subtracting the training mean.

    n_components must be an integer in 1..min(n_samples, n_features).
    components_ stores k right singular vectors as rows; singular_values_
    stores their singular values. explained_variance_ uses population variance
    of projected training samples (ddof=0), and explained_variance_ratio_
    divides it by the sum of original feature population variances, matching
    sklearn TruncatedSVD. This is not the uncentered squared-energy ratio.
    Ratios are defined as zeros when all training features have zero variance.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def _check_fitted(self):
        """Require a successful fit before projection or reconstruction."""
        if not hasattr(self, "components_"):
            raise RuntimeError("ManualSVDReducer is not fitted; call fit(X_train) first.")

    def fit(self, X):
        """Learn right singular vectors and variance statistics on training X."""
        matrix = as_real_matrix(X, "X")
        n_samples, n_features = matrix.shape
        k = self.n_components
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
            raise ValueError("n_components must be an integer.")
        upper = min(n_samples, n_features)
        if not 1 <= k <= upper:
            raise ValueError(
                f"n_components must be between 1 and min(n_samples, n_features) "
                f"= {upper}; got {k}."
            )
        k = int(k)
        # Economy SVD: X = U @ diag(s) @ Vt. Projection X @ V_k equals U_k S_k.
        _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        components = vt[:k].copy()
        rows = np.arange(k)
        pivots = np.argmax(np.abs(components), axis=1)
        components *= np.where(components[rows, pivots] < 0, -1.0, 1.0)[:, None]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                scores = matrix @ components.T
                variance = np.var(scores, axis=0, ddof=0)
                total_variance = np.var(matrix, axis=0, ddof=0).sum()
                ratios = variance / total_variance if total_variance > 0 else np.zeros(k)
        except FloatingPointError as exc:
            raise ValueError("X is too large for finite SVD statistics; rescale the values.") from exc
        if not np.isfinite(singular_values).all():
            raise ValueError("X is too large for finite singular values; rescale the values.")

        self.components_ = components
        self.singular_values_ = singular_values[:k].copy()
        self.explained_variance_ = variance
        self.explained_variance_ratio_ = ratios
        self.cumulative_explained_variance_ = np.cumsum(ratios)
        self.n_components_ = k
        self.n_features_in_ = n_features
        self.n_samples_ = n_samples
        return self

    def transform(self, X):
        """Project new samples using the training axes, without centering."""
        self._check_fitted()
        matrix = as_real_matrix(X, "X")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {matrix.shape[1]} features; expected {self.n_features_in_}."
            )
        return matrix @ self.components_.T

    def fit_transform(self, X):
        """Fit on X and return its uncentered component scores."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Reconstruct the rank-k approximation without adding a mean."""
        self._check_fitted()
        matrix = as_real_matrix(Z, "Z")
        if matrix.shape[1] != self.n_components_:
            raise ValueError(
                f"Z has {matrix.shape[1]} components; expected {self.n_components_}."
            )
        return matrix @ self.components_

