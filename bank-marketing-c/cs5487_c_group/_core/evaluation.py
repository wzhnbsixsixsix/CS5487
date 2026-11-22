"""Binary metrics with class 1 positive and continuous scores for AUC.

pr_auc is trapezoidal area under sklearn's precision-recall curve. It uses
auc(recall, precision); it is not average_precision_score (AP).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from ._validation import as_binary_labels, as_real_matrix


def _class_order(model):
    """Read binary class order; a model without classes_ assumes [0, 1]."""
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if classes.shape != (2,) or not np.isin(classes, [0, 1]).all() or len(np.unique(classes)) != 2:
        raise ValueError("model.classes_ must contain the two classes 0 and 1.")
    return int(np.flatnonzero(classes == 1)[0])


def _numeric_scores(values, n_samples):
    """Require a finite real numeric score vector for each evaluated sample."""
    raw = np.asarray(values)
    if raw.dtype.kind not in "biuf" or not np.isfinite(raw).all():
        raise ValueError("Model scores must contain finite real numeric values.")
    if raw.shape != (n_samples,):
        raise ValueError(f"Model scores must have shape ({n_samples},).")
    return raw.astype(np.float64)


def get_positive_class_scores(model, X):
    """Extract class-1 probabilities or oriented binary decision margins.

    Respect classes_ ordering. If absent, the interface contract assumes [0, 1]:
    probability column 1 or a positive decision margin corresponds to class 1.
    Do not silently substitute predicted labels for continuous scores.
    """
    matrix = as_real_matrix(X, "X")
    positive = _class_order(model)
    if callable(getattr(model, "predict_proba", None)):
        probabilities = np.asarray(model.predict_proba(matrix))
        if probabilities.shape != (len(matrix), 2):
            raise ValueError("predict_proba must return shape (n_samples, 2).")
        for column in range(2):
            _numeric_scores(probabilities[:, column], len(matrix))
        if np.any((probabilities < 0) | (probabilities > 1)) or not np.allclose(
            probabilities.sum(axis=1), 1, rtol=0, atol=1e-6
        ):
            raise ValueError("predict_proba must return probabilities in [0, 1] with row sums 1.")
        return probabilities[:, positive].astype(np.float64)
    if callable(getattr(model, "decision_function", None)):
        scores = _numeric_scores(model.decision_function(matrix), len(matrix))
        # Binary sklearn margins correspond to classes_[1].
        return scores if positive == 1 else -scores
    raise ValueError("Classifier needs predict_proba or decision_function for ROC-AUC/PR-AUC.")


def evaluate_classifier(model, X, y):
    """Evaluate an already-fitted binary model without fitting or mutating it.

    y must contain both 0 and 1 so both AUC metrics are defined. Undefined
    precision/recall/F1 due to absent positive predictions use zero_division=0.
    Model predictions define the classification threshold; AUC uses scores.
    """
    matrix = as_real_matrix(X, "X")
    labels = as_binary_labels(y, "y", len(matrix))
    if np.unique(labels).size != 2:
        raise ValueError("y must contain both classes 0 and 1 for ROC-AUC/PR-AUC evaluation.")
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Classifier must implement predict.")
    predicted = as_binary_labels(model.predict(matrix), "predictions", len(matrix))
    scores = get_positive_class_scores(model, matrix)
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(auc(recall, precision)),
    }

