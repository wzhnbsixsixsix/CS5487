"""Public orchestration for already-preprocessed group data; no implicit I/O."""
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from ._core.data_interface import prepare_experiment_data
from ._core.misclassification import get_misclassified_samples, summarize_misclassifications
from ._experiments.experiment_comparison import compare_reduction_methods
from ._experiments.experiment_k_selection import evaluate_pca_over_k

def _misclassification_tables(results, X_test, y_test, feature_names, data_source):
    """Reuse comparison models to analyze errors in the input feature space."""
    detail_tables, count_rows, feature_tables = [], [], []
    for method, result in results.items():
        model_input = X_test if result.reducer is None else result.reducer.transform(X_test)
        predicted = result.model.predict(model_input)
        details = get_misclassified_samples(X_test, y_test, predicted, feature_names)
        summary = summarize_misclassifications(X_test, y_test, predicted, feature_names)
        for frame in (details, summary["feature_summary"]):
            frame.insert(0, "feature_space", "preprocessed_original_features")
            frame.insert(0, "data_source", data_source)
            frame.insert(0, "method", method)
        detail_tables.append(details)
        feature_tables.append(summary["feature_summary"])
        count_rows.append({"method": method, "data_source": data_source, **summary["counts"]})
    return {
        "misclassified_samples": pd.concat(detail_tables, ignore_index=True),
        "misclassification_summary": pd.DataFrame(count_rows),
        "misclassification_feature_summary": pd.concat(feature_tables, ignore_index=True),
    }


@dataclass
class AnalysisResult:
    """Tables and fitted objects from one fixed train/evaluation split."""
    comparison: pd.DataFrame
    k_results: pd.DataFrame
    fitted_results: dict
    misclassified_samples: pd.DataFrame
    error_summary: pd.DataFrame
    feature_summary: pd.DataFrame


def run_analysis(X_train, X_eval, y_train, y_eval, feature_names, *,
                 n_components=None, k_values=None, classifier_factory=None,
                 data_source='supplied_preprocessed', output_dir=None, make_plots=False):
    """Compare representations, sweep PCA k and describe errors on supplied data.

    No resplitting/standardization; no files/plots by default. Use validation
    data for k exploration, then a separate held-out test for final reporting.
    n_components=None means min(4, n_train, n_features), a demo budget, not an
    optimized choice. Pass output_dir to save CSVs, make_plots=True to add PNGs.
    The factory must create fresh equal-config classifiers for every fit.
    """
    if make_plots and output_dir is None:
        raise ValueError('make_plots=True requires an explicit output_dir.')
    train, evaluation, train_y, eval_y, names = prepare_experiment_data(
        X_train, X_eval, y_train, y_eval, feature_names)
    k = min(4, *train.shape) if n_components is None else n_components
    comparison, fitted = compare_reduction_methods(
        train, evaluation, train_y, eval_y, k, classifier_factory=classifier_factory,
        output_path=None, data_source=data_source, return_results=True)
    sweep = evaluate_pca_over_k(
        train, evaluation, train_y, eval_y, k_values, classifier_factory=classifier_factory,
        output_path=None, data_source=data_source)
    errors = _misclassification_tables(fitted, evaluation, eval_y, names, data_source)
    result = AnalysisResult(comparison, sweep, fitted, errors['misclassified_samples'],
                            errors['misclassification_summary'], errors['misclassification_feature_summary'])
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        tables = destination / 'tables'
        tables.mkdir(parents=True, exist_ok=True)
        frames = {'reduction_comparison': comparison, 'pca_k_results': sweep, **errors}
        artifacts = []
        for name, frame in frames.items():
            frame.to_csv(tables / (name + '.csv'), index=False, encoding='utf-8-sig')
            artifacts.append('tables/' + name + '.csv')
        if make_plots:
            from .plotting import (plot_explained_variance, plot_pca_loadings,
                plot_pc1_pc2, plot_performance_vs_k, plot_method_comparison)
            directory = destination / 'figures'
            pca = fitted['PCA'].reducer
            total = pca.eigenvalues_.sum()
            ratios = pca.eigenvalues_ / total if total else pca.eigenvalues_.copy()
            plots = [('explained_variance', plot_explained_variance, (ratios,)),
                     ('pca_loadings', plot_pca_loadings, (pca.components_, names)),
                     ('performance_vs_k', plot_performance_vs_k, (sweep,)),
                     ('method_comparison', plot_method_comparison, (comparison,))]
            if pca.n_components_ >= 2:
                plots.append(('pc1_pc2', plot_pc1_pc2, (pca.transform(evaluation), eval_y)))
            for name, function, args in plots:
                function(*args, output_path=directory / (name + '.png'), data_source=data_source)
                artifacts.append('figures/' + name + '.png')
        (destination / 'analysis_run.json').write_text(json.dumps({
            'data_source': data_source, 'n_train': len(train), 'n_eval': len(evaluation),
            'n_features': len(names), 'comparison_components': fitted['PCA'].dimensions,
            'k_values': sweep.k.tolist(), 'artifacts': artifacts,
            'pr_auc_definition': 'trapezoidal area under precision-recall curve'
        }, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
