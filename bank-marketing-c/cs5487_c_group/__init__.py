"""Public Group C API. Importing this package does not run experiments or plot."""
from ._core.pca_manual import ManualPCA
from ._core.svd_manual import ManualSVDReducer
from ._core.data_interface import load_experiment_data, prepare_experiment_data
from ._core.classifier_interface import create_baseline_classifier
from ._core.evaluation import evaluate_classifier, get_positive_class_scores
from ._core.misclassification import get_misclassified_samples, summarize_misclassifications
from ._experiments.runner import ExperimentResult, run_experiment
from ._experiments.experiment_comparison import compare_reduction_methods
from ._experiments.experiment_k_selection import evaluate_pca_over_k, generate_k_values
from .analysis import AnalysisResult, run_analysis
from .group_a import GroupAData, GroupAGaussianNB, load_group_a_data, run_group_a_experiment
__version__ = '0.3.0'
__all__ = ['ManualPCA', 'ManualSVDReducer', 'load_experiment_data',
    'prepare_experiment_data', 'create_baseline_classifier', 'evaluate_classifier',
    'get_positive_class_scores', 'get_misclassified_samples', 'summarize_misclassifications',
    'ExperimentResult', 'run_experiment', 'compare_reduction_methods',
    'evaluate_pca_over_k', 'generate_k_values', 'AnalysisResult', 'run_analysis',
    'GroupAData', 'GroupAGaussianNB', 'load_group_a_data', 'run_group_a_experiment']
