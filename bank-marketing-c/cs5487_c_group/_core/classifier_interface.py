"""Classifier construction boundary for temporary and Group B models."""

from copy import deepcopy
from numbers import Integral

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE


def create_baseline_classifier():
    """Create a fresh, unfitted reproducible logistic regression baseline."""
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def create_classifier(classifier_factory=None):
    """Create an unfitted model via a factory (default: temporary baseline).

    Group B can supply a zero-argument callable returning a new estimator for
    each experiment. Estimators need fit, predict and either predict_proba or
    decision_function. Factory freshness/unfitted state is the provider's duty.
    """
    factory = create_baseline_classifier if classifier_factory is None else classifier_factory
    if not callable(factory):
        raise ValueError("classifier_factory must be a callable returning a fresh model.")
    model = factory()
    if not all(callable(getattr(model, name, None)) for name in ("fit", "predict")):
        raise ValueError("Classifier must implement fit and predict.")
    if not any(callable(getattr(model, name, None)) for name in ("predict_proba", "decision_function")):
        raise ValueError("Classifier must implement predict_proba or decision_function for AUC.")
    return model


def create_classifiers(n_models, *, classifier=None, classifier_factory=None):
    """Create independent models with the same configuration for a sweep.

    A sklearn prototype is cloned; an unfitted custom prototype is deep-copied.
    A factory must return a fresh, unfitted model with the same configuration
    each time. The supplied prototype is never fitted by this function.
    """
    if not isinstance(n_models, Integral) or isinstance(n_models, bool) or n_models < 1:
        raise ValueError("n_models must be a positive integer.")
    if classifier is not None and classifier_factory is not None:
        raise ValueError("Supply either classifier or classifier_factory, not both.")
    models = []
    for _ in range(n_models):
        if classifier is not None:
            candidate = clone(classifier) if callable(getattr(classifier, "get_params", None)) else deepcopy(classifier)
            if candidate is classifier:
                raise ValueError("Classifier prototype must support an independent clone or deepcopy.")
            model = create_classifier(lambda: candidate)
        else:
            model = create_classifier(classifier_factory)
        if any(model is previous for previous in models):
            raise ValueError("classifier_factory must return a fresh model for every experiment.")
        models.append(model)
    return models

