"""Figures for the aligned C experiment; all fit/selection happens upstream."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'Original': '#346b9c', 'PCA': '#c16a2b', 'SVD': '#298575'}


def _save(fig, out, name, note):
    fig.text(.5, .015, note, ha='center', fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(out/(name+'.png'), dpi=220)
    plt.close(fig)


def make_figures(out, data, pca, sweep, comparison, curves, ablations, selected):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlesize': 13}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for method in ('PCA', 'SVD'):
            rows = sweep.loc[sweep.method == method]
            axes[0].plot(rows.k, rows.pr_auc, '-o', ms=4, color=COLORS[method], label=method)
            axes[0].scatter(selected[method], rows.loc[rows.k == selected[method], 'pr_auc'],
                            s=100, facecolors='none', edgecolors=COLORS[method], linewidths=2)
            axes[1].plot(rows.k, rows.cumulative_explained_variance, '-o', ms=4,
                         color=COLORS[method], label=method)
        axes[0].set(xlabel='Number of components k', ylabel='Trapezoidal PR-AUC',
                    title='Select k independently on validation')
        axes[1].set(xlabel='Number of components k', ylabel='Retained variance fraction',
                    title='Variance is measured on training data')
        for ax in axes:
            ax.legend(frameon=False)
            ax.grid(alpha=.2)
        _save(fig, out, 'performance_vs_k', 'Bank-full | A fixed split | circles mark selected k | ties within 1e-12: smaller k')

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        x = np.arange(len(comparison))
        for ax, metric, title in zip(axes, ('pr_auc', 'f1'),
                                     ('Score ranking on held-out test', 'Test F1 at validation-selected cutoffs')):
            bars = ax.bar(x, comparison[metric], color=[COLORS[n] for n in comparison.method], width=.6)
            ax.bar_label(bars, fmt='%.4f', padding=4)
            ax.set_xticks(x, [f'{r.method}\n{int(r.dimensions)} features' for r in comparison.itertuples()])
            ax.set(ylabel=metric.replace('_', '-').upper(), ylim=(0, .55), title=title)
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
        _save(fig, out, 'method_comparison', 'Same A Gaussian NB (relative smoothing 0.01) | PCA/SVD k and cutoffs selected on validation')

        fig, ax = plt.subplots(figsize=(7.7, 4.8))
        for name, curve in curves.items():
            ax.plot(curve['recall'], curve['precision'], color=COLORS[name],
                    label=f'{name}: PR-AUC {curve["pr_auc"]:.4f}')
        ax.axhline(data.y_test.mean(), linestyle='--', color='#64748b', label=f'Prevalence {data.y_test.mean():.3f}')
        ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02), title='Held-out test precision-recall curves')
        ax.legend(frameon=False)
        _save(fig, out, 'test_pr_curves', 'Native log-odds scores | trapezoidal PR-AUC, distinct from average precision')

        fig, ax = plt.subplots(figsize=(8, 4.6))
        ratios = pca.eigenvalues_ / pca.eigenvalues_.sum()
        ax.plot(np.arange(1, len(ratios)+1), np.cumsum(ratios), '-o', ms=3, color=COLORS['PCA'])
        ax.axvline(int(np.sum(pca.eigenvalues_ > 1e-10)), color='#64748b', linestyle='--', label='Centered numerical rank')
        ax.set(xlabel='Number of principal components', ylabel='Cumulative explained variance',
               ylim=(0, 1.05), title='Training variance and redundant one-hot directions')
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        _save(fig, out, 'explained_variance', 'PCA eigenvalue cutoff for rank: 1e-10 | retaining all variance does not preserve NB decisions')

        if ablations is not None:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            names = ablations.representation.drop_duplicates().tolist()
            x = np.arange(len(names))
            for offset, mode, label, color in ((-.18, 'relative_0.01', 'Relative smoothing 0.01', '#c16a2b'),
                (.18, 'fixed_original_epsilon', 'Fixed Original absolute epsilon', '#346b9c')):
                rows = ablations.loc[ablations.smoothing_policy == mode].set_index('representation').loc[names]
                bars = ax.bar(x+offset, rows.pr_auc, .34, color=color, label=label)
                ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
            ax.set_xticks(x, [name.replace(' ', '\n') for name in names])
            ax.set(ylabel='Validation PR-AUC', ylim=(0, .48), title='Coordinate change and smoothing controls')
            ax.legend(frameon=False, loc='upper center', ncols=2)
            _save(fig, out, 'smoothing_ablation', 'Validation diagnostics only | full = all 50 axes | rank = all nonzero training directions')

        # Compact loadings keep feature labels readable in a report.
        components = pca.components_[:4]
        top = np.argsort(np.max(abs(components), axis=0))[-12:][::-1]
        fig, ax = plt.subplots(figsize=(11, 4.1))
        heat = ax.imshow(components[:, top], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(np.arange(len(top)), [data.feature_names[i] for i in top], rotation=35, ha='right')
        ax.set_yticks(range(4), ['PC1', 'PC2', 'PC3', 'PC4'])
        ax.set_title('Top 12 features by largest absolute coefficient in PC1-PC4')
        fig.colorbar(heat, ax=ax, label='Axis coefficient')
        _save(fig, out, 'pca_loadings', 'Training-fitted axes | sign convention is arbitrary | coefficients do not imply causation')

        scores = pca.transform(data.X_test)
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, color in ((0, '#346b9c'), (1, '#c16a2b')):
            mask = data.y_test == label
            ax.scatter(scores[mask, 0], scores[mask, 1], s=7, alpha=.35, color=color,
                       label=f'y={label} (n={int(mask.sum())})', rasterized=True)
        ax.set(xlabel='PC1', ylabel='PC2', title='Test samples in the first two training PCA axes')
        ax.legend(frameon=False)
        _save(fig, out, 'pc1_pc2', 'Descriptive projection only | overlap in two dimensions does not establish class inseparability')
