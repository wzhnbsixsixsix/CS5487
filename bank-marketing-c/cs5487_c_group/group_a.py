"""Integration with the unchanged bank-marketing-a module in the shared repo."""
from dataclasses import dataclass
from pathlib import Path
import csv
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd

from ._core._validation import as_binary_labels


def _load_a(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[2] / 'bank-marketing-a/bank_marketing_a.py'
    if not path.is_file():
        raise FileNotFoundError('A module not found; supply a_module_path pointing to bank_marketing_a.py.')
    spec = importlib.util.spec_from_file_location('_cs5487_external_group_a', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('load_bank_data', 'stratified_split', 'BankPreprocessor', 'GaussianNB'):
        if not callable(getattr(module, name, None)):
            raise ValueError(f'A module is missing {name}.')
    return module, path.resolve()


@dataclass
class GroupAData:
    """The original A split and its training-fitted preprocessing."""
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: list
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    preprocessor: object
    a_module: object
    a_module_path: Path
    split_path: object
    rows: list


def _read_split(path, y):
    buckets = {name: [] for name in ('train', 'validation', 'test')}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'split', 'source_record', 'y'}.issubset(reader.fieldnames or []):
            raise ValueError('split_indices.csv must contain split, source_record and y.')
        for row in reader:
            if row['split'] not in buckets:
                raise ValueError('Unknown split name in split_indices.csv.')
            index, label = int(row['source_record']) - 1, int(row['y'])
            if not 0 <= index < len(y) or label != int(y[index]):
                raise ValueError('Split record/label does not match the supplied dataset.')
            buckets[row['split']].append(index)
    all_indices = sum(buckets.values(), [])
    if len(all_indices) != len(y) or len(set(all_indices)) != len(y):
        raise ValueError('Split must cover every record exactly once without overlap.')
    parts = tuple(np.asarray(values, dtype=int) for values in buckets.values())
    if any(set(y[indices]) != {0, 1} for indices in parts):
        raise ValueError('Each split must contain both classes.')
    return parts


def load_group_a_data(data_path, *, a_module_path=None, split_path=None, seed=42,
                      include_duration=False):
    """Reuse A's CSV/ZIP reader, exact split and training-only preprocessor.

    Pass A's split_indices.csv to reproduce its published records/order. If
    omitted, call A.stratified_split(seed), never sklearn's alternative split.
    Numeric columns alone are standardized by A; one-hot columns stay 0/1.
    """
    module, module_path = _load_a(a_module_path)
    rows, y = module.load_bank_data(data_path)
    y = as_binary_labels(y, 'y', len(rows))
    indices = _read_split(split_path, y) if split_path is not None else module.stratified_split(y, seed=seed)
    train, validation, test = indices
    prep = module.BankPreprocessor(include_duration=include_duration).fit([rows[i] for i in train])
    matrices = [prep.transform([rows[i] for i in part]) for part in indices]
    return GroupAData(*matrices, y[train], y[validation], y[test], list(prep.feature_names),
                      train, validation, test, prep, module, module_path,
                      Path(split_path).resolve() if split_path is not None else None, rows)


class GroupAGaussianNB:
    """Adapter exposing A's native log-odds, avoiding saturated probability ties.

    Starts at A's default threshold 0. The aligned experiment sets the
    validation-selected cutoff after fitting; ranking scores stay unchanged.
    """
    def __init__(self, a_module_path=None, var_smoothing=0.01, log_odds_threshold=0.0):
        self.a_module_path = a_module_path
        self.var_smoothing = var_smoothing
        self.log_odds_threshold = log_odds_threshold

    def fit(self, X, y):
        module, _ = _load_a(self.a_module_path)
        self.model_ = module.GaussianNB(var_smoothing=self.var_smoothing).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError('GroupAGaussianNB is not fitted.')
        return self.model_.decision_function(X)

    def predict(self, X):
        return (self.decision_function(X) >= self.log_odds_threshold).astype(int)


def _run_group_a_experiment_legacy(data_path, *, a_module_path=None, split_path=None,
                           output_dir='results_c', seed=42, include_duration=False,
                           k_values=None, select_metric='pr_auc', classifier_factory=None):
    """Select k on A's validation split, then compare on its untouched test split.

    All fits use A's training rows only; there is no train+validation refit.
    Ties in the selected metric favor smaller k. Custom factories replace
    A's NB consistently in both validation and test experiments.
    """
    from ._experiments.experiment_k_selection import evaluate_pca_over_k, generate_k_values
    from ._experiments.experiment_comparison import compare_reduction_methods
    from .analysis import _misclassification_tables
    from .plotting import (plot_explained_variance, plot_pca_loadings, plot_pc1_pc2,
                           plot_performance_vs_k, plot_method_comparison)
    if select_metric not in ('pr_auc', 'f1', 'roc_auc'):
        raise ValueError('select_metric must be pr_auc, f1 or roc_auc.')
    data = load_group_a_data(data_path, a_module_path=a_module_path, split_path=split_path,
                             seed=seed, include_duration=include_duration)
    factory = classifier_factory if classifier_factory is not None else lambda: GroupAGaussianNB(data.a_module_path)
    candidates = generate_k_values(*data.X_train.shape, step=5) if k_values is None else k_values
    validation = evaluate_pca_over_k(data.X_train, data.X_validation, data.y_train,
                                     data.y_validation, candidates, classifier_factory=factory,
                                     output_path=None, data_source='group_a_bank_full')
    validation['evaluation_split'] = 'validation'
    selected = validation.sort_values([select_metric, 'k'], ascending=[False, True]).iloc[0]
    k = int(selected.k)
    comparison, fitted = compare_reduction_methods(data.X_train, data.X_test, data.y_train,
        data.y_test, k, classifier_factory=factory, output_path=None,
        data_source='group_a_bank_full', return_results=True)
    comparison['evaluation_split'] = 'test'
    errors = _misclassification_tables(fitted, data.X_test, data.y_test,
                                       data.feature_names, 'group_a_bank_full')
    details = errors['misclassified_samples']
    details.insert(0, 'source_record', data.test_indices[details.row_position.to_numpy(dtype=int)] + 1)
    tables = {'validation_k_results': validation, 'reduction_comparison': comparison, **errors}
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / (name + '.csv'), index=False, encoding='utf-8-sig')
    split = pd.DataFrame([{'split': name, 'source_record': int(i)+1, 'y': int(y)}
        for name, part, labels in zip(('train','validation','test'),
            (data.train_indices, data.validation_indices, data.test_indices),
            (data.y_train, data.y_validation, data.y_test)) for i,y in zip(part, labels)])
    split.to_csv(out / 'split_indices.csv', index=False)
    pca = fitted['PCA'].reducer
    total = pca.eigenvalues_.sum()
    source = 'group_a_bank_full | validation-selected k'
    plot_explained_variance(pca.eigenvalues_/total if total else pca.eigenvalues_,
        output_path=out/'explained_variance.png', data_source=source)
    plot_pca_loadings(pca.components_[:4], data.feature_names,
        output_path=out/'pca_loadings.png', data_source=source)
    plot_performance_vs_k(validation, output_path=out/'performance_vs_k.png', data_source='group_a_bank_full | VALIDATION')
    plot_method_comparison(comparison, output_path=out/'method_comparison.png', data_source='group_a_bank_full | TEST')
    if k >= 2:
        plot_pc1_pc2(pca.transform(data.X_test), data.y_test, output_path=out/'pc1_pc2.png', data_source='group_a_bank_full | TEST')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    summary = {'data_file': Path(data_path).name, 'data_sha256': digest(data_path),
        'a_module_sha256': digest(data.a_module_path), 'split_indices_sha256': digest(out/'split_indices.csv'),
        'provided_split_sha256': digest(data.split_path) if data.split_path is not None else None,
        'k_candidates': validation.k.tolist(),
        'split_source': 'provided_A_csv' if split_path is not None else 'A_stratified_split',
        'seed': seed if split_path is None else None,
        'include_duration': include_duration, 'n_features': len(data.feature_names),
        'feature_names': data.feature_names, 'n_train': len(data.y_train),
        'n_validation': len(data.y_validation), 'n_test': len(data.y_test),
        'selected_k': k, 'selection_metric': select_metric, 'selection_split': 'validation',
        'validation_selection_value': float(selected[select_metric]),
        'threshold_policy': 'A GaussianNB log-odds >= 0' if classifier_factory is None else 'custom model.predict',
        'classifier': 'A handwritten GaussianNB; var_smoothing=0.01' if classifier_factory is None else 'custom factory',
        'preprocessing': 'unchanged A preprocessing; numeric-only Z-score, one-hot unchanged',
        'test_metrics': comparison.to_dict(orient='records'),
        'pr_auc': 'trapezoidal area using native continuous scores; not AP',
        'training_policy': 'training split only; no train+validation refit'}
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return {'tables': tables, 'metrics': summary, 'fitted_results': fitted, 'data': data}


def run_group_a_experiment(data_path, *, a_module_path=None, split_path=None,
                           output_dir='results_c', seed=42, include_duration=False,
                           k_values=None, select_metric='pr_auc', classifier_factory=None,
                           b_metrics_path=None, make_plots=True):
    """A/B-aligned NB protocol; custom factories retain the older generic API.

    The standard NB run independently selects PCA/SVD k and score cutoffs on
    validation, exports both threshold policies and performs diagnostic controls.
    For arbitrary classifiers use run_analysis for an explicitly supplied split.
    """
    if classifier_factory is not None:
        if b_metrics_path is not None or not make_plots:
            raise ValueError('Custom factories use the legacy protocol without b_metrics_path/make_plots options.')
        return _run_group_a_experiment_legacy(data_path, a_module_path=a_module_path,
            split_path=split_path, output_dir=output_dir, seed=seed,
            include_duration=include_duration, k_values=k_values,
            select_metric=select_metric, classifier_factory=classifier_factory)
    from .aligned_experiment import run_aligned_experiment
    return run_aligned_experiment(data_path, a_module_path=a_module_path,
        split_path=split_path, output_dir=output_dir, seed=seed,
        include_duration=include_duration, k_values=k_values, select_metric=select_metric,
        b_metrics_path=b_metrics_path, make_plots=make_plots)
