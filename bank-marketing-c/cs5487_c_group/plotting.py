"""Optional explicit plotting imports; backend selection belongs to the caller."""
from ._core.visualization import (plot_explained_variance, plot_pca_loadings,
    plot_pc1_pc2, plot_performance_vs_k, plot_method_comparison)
__all__ = ['plot_explained_variance', 'plot_pca_loadings', 'plot_pc1_pc2',
           'plot_performance_vs_k', 'plot_method_comparison']
