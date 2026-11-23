"""C's reproducible representation study, using A's unchanged score protocol.

All representation/classifier fits use training data. Component counts and
decision cutoffs use validation data. Diagnostic ablations use validation only.
B's saved results are read only after their data/split/baseline checks pass.
"""
from copy import copy
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from ._core.pca_manual import ManualPCA
from ._core.svd_manual import ManualSVDReducer
from ._core.evaluation import get_positive_class_scores
from ._experiments.runner import ExperimentResult
from .group_a import GroupAGaussianNB, load_group_a_data

TIE_ATOL = 1e-12
POLICY = 'validation_selected'


def choose_k(table, metric='pr_auc'):
    """Maximize a validation metric; numerical ties prefer the smaller k."""
    best = table[metric].max()
    return int(table.loc[table[metric] >= best - TIE_ATOL, 'k'].min())


def evaluate_scores(a, labels, scores, threshold):
    curves = a.ranking_curves(labels, scores)
    matrix = a.confusion_matrix(labels, (scores >= threshold).astype(int))
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {**a.classification_metrics(matrix),
            **{key: float(curves[key]) for key in ('pr_auc', 'roc_auc', 'average_precision')},
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'log_odds_threshold': float(threshold)}


def _cutoff(a, labels, scores):
    return a.select_threshold(a.ranking_curves(labels, scores))


def _prefix(reducer, k):
    """Slice one training-fitted basis; do not refit for every prefix."""
    result = copy(reducer)
    result.components_ = reducer.components_[:k].copy()
    result.n_components_ = k
    for name in ('singular_values_', 'explained_variance_',
                 'explained_variance_ratio_', 'cumulative_explained_variance_'):
        if hasattr(reducer, name):
            setattr(result, name, getattr(reducer, name)[:k].copy())
    return result


def _candidates(shape, k_values, ranks):
    upper = min(shape)
    if k_values is None:
        return sorted({1, upper, *range(5, upper + 1, 5),
                       *(k for rank in ranks for k in (rank-1, rank, rank+1) if 1 <= k <= upper)})
    values = list(k_values)
    if not values or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral)
                         or not 1 <= k <= upper for k in values):
        raise ValueError(f'k_values must contain integers in 1..{upper}.')
    return sorted(set(map(int, values)))


def _off_diagonal_fraction(matrix):
    covariance = np.cov(matrix, rowvar=False)
    covariance = np.atleast_2d(covariance)
    total = float(np.square(covariance).sum())
    off = total - float(np.square(np.diag(covariance)).sum())
    return max(0., off / total) if total else 0.


def _canonical_digest(data):
    digest = hashlib.sha256()
    labels = np.empty(len(data.rows), dtype=int)
    for indices, y in zip((data.train_indices, data.validation_indices, data.test_indices),
                          (data.y_train, data.y_validation, data.y_test)):
        labels[indices] = y
    for row, label in zip(data.rows, labels):
        digest.update((json.dumps([row, int(label)], ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':')) + '\n').encode('utf-8'))
    return digest.hexdigest()


def _team_table(data, main_table, canonical_digest, b_metrics_path):
    """Check provenance before incorporating existing A/B model metrics."""
    if b_metrics_path is None:
        return None, {'status': 'not_requested'}
    path = Path(b_metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f'B metrics not found: {path}')
    b = json.loads(path.read_text(encoding='utf-8'))
    if ('duration' in data.feature_names) != b['include_duration'] or canonical_digest != b['data']['canonical_records_sha256']:
        raise ValueError('B results do not match C input data/duration setting.')
    expected_split = data.a_module.stratified_split(_labels_in_source_order(data), seed=b['seed'])
    if any(not np.array_equal(actual, expected) for actual, expected in zip(
        (data.train_indices, data.validation_indices, data.test_indices), expected_split)):
        raise ValueError('B results do not match C split records/order.')
    baseline = main_table.loc[main_table.method == 'Original'].iloc[0]
    saved = b['test']['Gaussian NB']
    for key in ('pr_auc', 'roc_auc', 'average_precision'):
        if not np.isclose(baseline[key], saved['ranking'][key], rtol=0, atol=1e-10):
            raise ValueError(f'C original NB differs from B baseline: {key}.')
    saved_cutoff = saved[POLICY]['log_odds_threshold']
    if not np.isclose(baseline.log_odds_threshold, saved_cutoff, rtol=0, atol=1e-10):
        raise ValueError('C original NB threshold differs from A/B.')
    if [int(baseline[k]) for k in ('tn', 'fp', 'fn', 'tp')] != list(np.asarray(saved[POLICY]['confusion_matrix']).ravel()):
        raise ValueError('C original NB confusion matrix differs from A/B.')
    rows = []
    for name in ('MLE', 'MAP'):
        source = b['test'][name]
        metrics = source[POLICY]
        tn, fp, fn, tp = map(int, np.asarray(metrics['confusion_matrix']).ravel())
        rows.append({'method': name, 'dimensions': b['n_logistic_features'],
                     **source['ranking'], **{key: metrics[key] for key in
                     ('accuracy', 'precision', 'recall', 'f1', 'log_odds_threshold')},
                     'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                     'threshold_policy': POLICY, 'evaluation_split': 'test',
                     'provenance': 'B saved metrics; no B retraining'})
    own = main_table.copy()
    own['method'] = own.method.replace({'Original': 'Gaussian NB', 'PCA': 'PCA + NB', 'SVD': 'SVD + NB'})
    own['provenance'] = 'C rerun using unchanged A NB'
    return pd.concat([own.iloc[:1], pd.DataFrame(rows), own.iloc[1:]], ignore_index=True), {
        'status': 'matched', 'path': str(path.resolve()),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope': 'data fingerprint, exact seed-42 split/order and NB baseline; B optimizer/CV not rerun'}


def _labels_in_source_order(data):
    labels = np.empty(len(data.rows), dtype=int)
    for indices, y in zip((data.train_indices, data.validation_indices, data.test_indices),
                          (data.y_train, data.y_validation, data.y_test)):
        labels[indices] = y
    return labels


def run_aligned_experiment(data_path, *, a_module_path=None, split_path=None,
                           output_dir='results_c', seed=42, include_duration=False,
                           k_values=None, select_metric='pr_auc', classifier_factory=None,
                           b_metrics_path=None, make_plots=True):
    if classifier_factory is not None:
        raise ValueError('The aligned protocol uses A GaussianNB; use run_analysis for custom classifiers.')
    if select_metric not in ('pr_auc', 'roc_auc', 'f1'):
        raise ValueError('select_metric must be pr_auc, roc_auc or f1.')
    data = load_group_a_data(data_path, a_module_path=a_module_path, split_path=split_path,
                             seed=seed, include_duration=include_duration)
    a = data.a_module
    train, validation, test = data.X_train, data.X_validation, data.X_test
    upper = min(train.shape)
    pca = ManualPCA(upper).fit(train)
    svd = ManualSVDReducer(upper).fit(train)
    centered_rank = int(np.sum(pca.eigenvalues_ > 1e-10))
    raw_rank = int(np.linalg.matrix_rank(train))
    candidates = _candidates(train.shape, k_values, (centered_rank, raw_rank))
    bases = {'PCA': pca, 'SVD': svd}
    projected = {name: (basis.transform(train), basis.transform(validation))
                 for name, basis in bases.items()}
    factory = classifier_factory or (lambda: GroupAGaussianNB(data.a_module_path))
    seen_models = []

    def fit_model(matrix, *, absolute_epsilon=None):
        model = factory()
        if any(model is old for old in seen_models):
            raise ValueError('classifier_factory must return a fresh model for each fit.')
        seen_models.append(model)
        if absolute_epsilon is not None:
            # A uses this exact population variance expression to set epsilon.
            center = matrix.sum(axis=0) / len(matrix)
            scale = max(float((((matrix-center)**2).sum(axis=0)/len(matrix)).max()), 1e-12)
            model.var_smoothing = absolute_epsilon / scale
        return model.fit(matrix, data.y_train)

    validation_rows, fitted_candidates = [], {}
    for name, (training_scores, validation_scores) in projected.items():
        for k in candidates:
            model = fit_model(training_scores[:, :k])
            scores = get_positive_class_scores(model, validation_scores[:, :k])
            threshold = _cutoff(a, data.y_validation, scores)
            row = {'method': name, 'k': k, **evaluate_scores(a, data.y_validation, scores, threshold),
                   'default_f1': evaluate_scores(a, data.y_validation, scores, 0.)['f1'],
                   'cumulative_explained_variance': float(bases[name].cumulative_explained_variance_[k-1]),
                   'evaluation_split': 'validation', 'threshold_policy': POLICY}
            validation_rows.append(row)
            fitted_candidates[name, k] = (model, threshold)
    sweep = pd.DataFrame(validation_rows)
    selected = {name: choose_k(sweep.loc[sweep.method == name], select_metric) for name in bases}
    original = fit_model(train)
    original_threshold = _cutoff(a, data.y_validation, get_positive_class_scores(original, validation))
    # Selection is frozen above. No test score or label participates in it.
    comparison_rows, defaults, fitted, predictions = [], [], {}, []
    test_curves = {}
    for name in ('Original', 'PCA', 'SVD'):
        k = train.shape[1] if name == 'Original' else selected[name]
        reducer = None if name == 'Original' else _prefix(bases[name], k)
        model, threshold = (original, original_threshold) if reducer is None else fitted_candidates[name, k]
        matrix = test if reducer is None else reducer.transform(test)
        scores = get_positive_class_scores(model, matrix)
        if isinstance(model, GroupAGaussianNB):
            model.log_odds_threshold = threshold
        tuned = evaluate_scores(a, data.y_test, scores, threshold)
        common = {'method': name, 'dimensions': k, 'evaluation_split': 'test',
                  'epsilon': model.model_.epsilon}
        comparison_rows.append({**common, **tuned, 'threshold_policy': POLICY})
        defaults.append({**common, **evaluate_scores(a, data.y_test, scores, 0.),
                         'threshold_policy': 'default_log_odds_0'})
        fitted[name] = ExperimentResult(model, reducer, tuned, k)
        test_curves[name] = a.ranking_curves(data.y_test, scores)
        predictions.append(pd.DataFrame({'method': name, 'source_record': data.test_indices+1,
            'y_true': data.y_test, 'log_odds': scores, 'log_odds_threshold': threshold,
            'y_pred': (scores >= threshold).astype(int), 'y_pred_default': (scores >= 0).astype(int),
            'threshold_policy': POLICY}))
    main_table, default_table = pd.DataFrame(comparison_rows), pd.DataFrame(defaults)
    fixed_rows, legacy_rows = [], []
    for k, target in ((selected['PCA'], fixed_rows), (min(45, upper), legacy_rows)):
        for name in ('PCA', 'SVD'):
            if (name, k) in fitted_candidates:
                model, cutoff = fitted_candidates[name, k]
            else:
                training_scores, validation_scores = projected[name]
                model = fit_model(training_scores[:, :k])
                cutoff = _cutoff(a, data.y_validation,
                                 get_positive_class_scores(model, validation_scores[:, :k]))
            scores = get_positive_class_scores(model, _prefix(bases[name], k).transform(test))
            for policy, threshold in ((POLICY, cutoff), ('default_log_odds_0', 0.)):
                target.append({'method': name, 'dimensions': k, 'threshold_policy': policy,
                               'evaluation_split': 'test', **evaluate_scores(a, data.y_test, scores, threshold)})

    # Fixed epsilon is computed from training Original, never chosen on test.
    ablations, diagnostics = [], []
    if classifier_factory is None:
        epsilon = original.model_.epsilon
        representations = {'Original': (train, validation),
            'Centered identity': (train-pca.mean_, validation-pca.mean_),
            'PCA full': projected['PCA'],
            'PCA rank': tuple(matrix[:, :max(1, centered_rank)] for matrix in projected['PCA']),
            'SVD full': projected['SVD'],
            'SVD rank': tuple(matrix[:, :max(1, raw_rank)] for matrix in projected['SVD'])}
        for name, (x_train, x_validation) in representations.items():
            for mode in ('relative_0.01', 'fixed_original_epsilon'):
                model = fit_model(x_train, absolute_epsilon=epsilon if mode.startswith('fixed') else None)
                scores = model.decision_function(x_validation)
                cutoff = _cutoff(a, data.y_validation, scores)
                ablations.append({'representation': name, 'dimensions': x_train.shape[1],
                    'smoothing_policy': mode, 'epsilon': model.model_.epsilon,
                    'var_smoothing': model.var_smoothing, 'evaluation_split': 'validation',
                    **evaluate_scores(a, data.y_validation, scores, cutoff)})
            diagnostics.append({'representation': name, 'dimensions': x_train.shape[1],
                'pooled_offdiagonal_squared_fraction': _off_diagonal_fraction(x_train),
                'class_0_offdiagonal_squared_fraction': _off_diagonal_fraction(x_train[data.y_train == 0]),
                'class_1_offdiagonal_squared_fraction': _off_diagonal_fraction(x_train[data.y_train == 1])})
    canonical = _canonical_digest(data)
    team_table, b_provenance = _team_table(data, main_table, canonical, b_metrics_path)
    from .analysis import _misclassification_tables
    if classifier_factory is None:
        errors = _misclassification_tables(fitted, test, data.y_test, data.feature_names, 'group_a_bank_full')
        details = errors['misclassified_samples']
        details.insert(0, 'source_record', data.test_indices[details.row_position.to_numpy(dtype=int)] + 1)
        for table in errors.values():
            table['threshold_policy'] = POLICY
    else:
        errors = {}
    prediction_table = pd.concat(predictions, ignore_index=True)
    tables = {'validation_k_results': sweep, 'reduction_comparison': main_table,
              'reduction_comparison_default': default_table,
              'fixed_k_comparison': pd.DataFrame(fixed_rows),
              'legacy_k45_comparison': pd.DataFrame(legacy_rows),
              'test_predictions': prediction_table, **errors}
    if ablations:
        tables['smoothing_ablation_validation'] = pd.DataFrame(ablations)
        tables['coordinate_diagnostics_train'] = pd.DataFrame(diagnostics)
    if team_table is not None:
        tables['team_comparison'] = team_table
    tables['qualitative_cases'] = qualitative_cases(data, prediction_table)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        # Keep full predictions and historical controls available to API callers.
        if name in ('test_predictions', 'legacy_k45_comparison'):
            continue
        table.to_csv(out/(name+'.csv'), index=False, encoding='utf-8-sig')
    split = pd.DataFrame([{'split': name, 'source_record': int(i)+1, 'y': int(y)}
        for name, indices, labels in zip(('train', 'validation', 'test'),
        (data.train_indices, data.validation_indices, data.test_indices),
        (data.y_train, data.y_validation, data.y_test)) for i, y in zip(indices, labels)])
    split.to_csv(out/'split_indices.csv', index=False)
    reconstruction = {}
    for name, basis in bases.items():
        for k in sorted({selected[name], upper}):
            reducer = _prefix(basis, k)
            reconstruction[f'{name}_{k}_train_relative_error'] = float(
                np.linalg.norm(reducer.inverse_transform(reducer.transform(train))-train) / np.linalg.norm(train))
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    summary = {'protocol_version': 'C-aligned-1', 'data_file': Path(data_path).name,
        'data_sha256': digest(data_path), 'canonical_records_sha256': canonical,
        'a_module_sha256': digest(data.a_module_path), 'split_indices_sha256': digest(out/'split_indices.csv'),
        'provided_split_sha256': digest(data.split_path) if data.split_path else None,
        'split_source': 'provided_A_csv' if split_path else 'A_stratified_split',
        'seed': seed if split_path is None else None, 'include_duration': include_duration,
        'n_features': len(data.feature_names), 'feature_names': data.feature_names,
        'n_train': len(train), 'n_validation': len(validation), 'n_test': len(test),
        'test_positives': int(data.y_test.sum()), 'k_candidates': candidates,
        'selected_k': selected['PCA'], 'selected_k_by_method': selected,
        'selection_metric': select_metric, 'selection_split': 'validation',
        'selection_tie_absolute_tolerance': TIE_ATOL,
        'threshold_policy': 'each representation: validation maximum F1; ties choose higher score cutoff',
        'default_threshold_output': 'reduction_comparison_default.csv; log-odds >= 0',
        'classifier': 'unchanged A handwritten GaussianNB; var_smoothing=0.01' if classifier_factory is None else 'custom factory; score cutoffs selected on validation',
        'preprocessing': 'unchanged A; numeric-only Z-score; complete one-hot (50 columns)',
        'training_policy': 'training split only; no train+validation refit',
        'pr_auc': 'trapezoidal native continuous-score PR-AUC, distinct from AP',
        'rank': {'centered_covariance_eigenvalue_gt_1e-10': centered_rank,
                 'uncentered_numpy_matrix_rank': raw_rank},
        'pca_variance_at_selected_k': float(pca.cumulative_explained_variance_[selected['PCA']-1]),
        'reconstruction': reconstruction, 'ablation_split': 'validation only',
        'b_provenance': b_provenance, 'test_metrics': main_table.to_dict(orient='records'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     **{name: version(name) for name in ('scipy', 'scikit-learn', 'matplotlib')}}}
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if make_plots:
        from .aligned_plots import make_figures
        make_figures(out, data, pca, sweep, main_table, test_curves, tables.get('smoothing_ablation_validation'), selected)
    from .reporting import write_results_report
    write_results_report(out, summary, tables, make_plots)
    return {'tables': tables, 'metrics': summary, 'fitted_results': fitted, 'data': data}


def qualitative_cases(data, predictions):
    """First two source records per transition; descriptive, never for tuning."""
    original = predictions.loc[predictions.method == 'Original'].set_index('source_record')
    rows = []
    for method in ('PCA', 'SVD'):
        reduced = predictions.loc[predictions.method == method].set_index('source_record')
        for label in (0, 1):
            for corrected in (True, False):
                mask = ((original.y_true == label) &
                        ((original.y_pred != label) if corrected else (original.y_pred == label)) &
                        ((reduced.y_pred == label) if corrected else (reduced.y_pred != label)))
                for record in sorted(original.index[mask])[:2]:
                    raw = {k: v for k, v in data.rows[record-1].items() if k != 'duration'}
                    rows.append({'method': method, 'source_record': record,
                        'transition': ('corrected_' if corrected else 'introduced_') + ('FP' if label == 0 else 'FN'),
                        **raw, 'y_true': label, 'original_pred': int(original.loc[record, 'y_pred']),
                        'reduced_pred': int(reduced.loc[record, 'y_pred']),
                        'original_log_odds': original.loc[record, 'log_odds'],
                        'reduced_log_odds': reduced.loc[record, 'log_odds'],
                        'original_threshold': original.loc[record, 'log_odds_threshold'],
                        'reduced_threshold': reduced.loc[record, 'log_odds_threshold']})
    return pd.DataFrame(rows)
