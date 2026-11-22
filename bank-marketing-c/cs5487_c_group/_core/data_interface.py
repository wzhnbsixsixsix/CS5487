"""Single data boundary: synthetic development data or Group A's existing split.

No experiment module should read CSVs. Future raw Bank Marketing loading belongs
here; existing bank/bank-additional files are not loaded by the default demo.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ._validation import as_binary_labels, as_real_matrix
from .config import RANDOM_STATE


def prepare_experiment_data(
    X_train, X_test, y_train, y_test, feature_names, *, standardize=False
):
    """Validate a provided split; optionally scale using training statistics only.

    Return dense numeric arrays and a copied feature-name list. Do not split,
    reorder columns or infer categorical encodings. DataFrame columns must
    match feature_names in order. Supplied preprocessed data is unchanged by
    default. Raw categorical CSVs must first be processed by the data provider.
    """
    if isinstance(feature_names, (str, bytes)):
        raise ValueError("feature_names must be a sequence of unique feature names.")
    try:
        names = list(feature_names)
    except TypeError as exc:
        raise ValueError("feature_names must be a sequence of feature names.") from exc
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("feature_names must contain nonempty strings.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    for frame, name in ((X_train, "X_train"), (X_test, "X_test")):
        if isinstance(frame, pd.DataFrame) and list(frame.columns) != names:
            raise ValueError(f"{name} DataFrame columns must match feature_names in order.")
    train = as_real_matrix(X_train, "X_train").copy()
    test = as_real_matrix(X_test, "X_test").copy()
    if train.shape[1] != test.shape[1] or train.shape[1] != len(names):
        raise ValueError("Train/test feature dimensions must match feature_names.")
    train_y = as_binary_labels(y_train, "y_train", len(train))
    test_y = as_binary_labels(y_test, "y_test", len(test))
    if np.unique(train_y).size != 2:
        raise ValueError("y_train must contain both classes 0 and 1.")
    if standardize:
        scaler = StandardScaler().fit(train)
        train, test = scaler.transform(train), scaler.transform(test)
        # Extreme finite inputs can overflow during preprocessing.
        train = as_real_matrix(train, "scaled X_train")
        test = as_real_matrix(test, "scaled X_test")
    return train, test, train_y, test_y, names


def load_experiment_data(*, preprocessed=None, random_state=RANDOM_STATE):
    """Return (X_train, X_test, y_train, y_test, feature_names).

    Default: 600 simulated samples, 12 features, stratified 75/25 split, then
    training-only standardization. A supplied five-item preprocessed tuple is
    validated and returned without resplitting or further standardization.
    Replace the provider here when Group A's actual processing is available.
    """
    if preprocessed is not None:
        if not isinstance(preprocessed, (tuple, list)) or len(preprocessed) != 5:
            raise ValueError("preprocessed must contain the five data-interface items.")
        return prepare_experiment_data(*preprocessed)
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, n_redundant=3,
        weights=[0.85, 0.15], random_state=random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state,
    )
    names = [f"synthetic_feature_{i + 1}" for i in range(X.shape[1])]
    return prepare_experiment_data(
        X_train, X_test, y_train, y_test, names, standardize=True,
    )

