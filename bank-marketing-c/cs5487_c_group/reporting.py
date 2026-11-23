"""Generate a concise result note from the current experiment tables."""


def markdown_table(frame, columns, labels=None):
    labels = labels or columns
    lines = ['| ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---']*len(columns)) + ' |']
    for _, row in frame.iterrows():
        cells = []
        for key in columns:
            value = row[key]
            if key in ('dimensions', 'k', 'fp', 'fn', 'source_record', 'y_true', 'original_pred', 'reduced_pred'):
                cells.append(str(int(value)))
            elif isinstance(value, (float, int)):
                cells.append(f'{value:.4f}')
            else:
                cells.append(str(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_results_report(out, summary, tables, make_plots):
    """Use run metadata so the note stays correct for custom data and grids."""
    selected = summary['selected_k_by_method']
    sweep = tables['validation_k_results']
    selected_rows = sweep.loc[[row.k == selected[row.method] for row in sweep.itertuples()]]
    duration = '包含 duration' if summary['include_duration'] else '排除 duration'
    columns = ['method', 'dimensions', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'fp', 'fn']
    text = f'''# PCA 与 SVD 实验结果

数据：{summary['data_file']}，{duration}；训练/验证/测试分别为 {summary['n_train']:,} / {summary['n_validation']:,} / {summary['n_test']:,}，测试正例 {summary['test_positives']}。
预处理使用训练集拟合的标准化与完整独热编码，共 {summary['n_features']} 列。

## 实验设置与 k 选择

分类器为 A 手写 Gaussian NB，var_smoothing=0.01。PCA 中心化后分解协方差矩阵，SVD 对未中心化矩阵做经济型分解。
两种方法分别按验证集 {summary['selection_metric']} 选 k，数值差不超过 {summary['selection_tie_absolute_tolerance']:g} 时取较小 k。
候选值为 {summary['k_candidates']}。每种表示再按验证集最大 F1 选阈值，并列取较高阈值，固定后评价测试集。
PR-AUC 为连续 log-odds 的梯形积分，AP 单独保存。

{markdown_table(selected_rows, ['method', 'k', 'pr_auc', 'cumulative_explained_variance', 'log_odds_threshold'], ['表示', 'k', '验证 PR-AUC', '累计方差比例', '验证阈值'])}

## 测试集主结果

各行使用自己的验证集阈值与所选维数。

{markdown_table(tables['reduction_comparison'], columns)}

固定相同 k={selected['PCA']} 的比较见 fixed_k_comparison.csv。

## 默认阈值辅助结果

以下统一使用 log-odds >= 0。排序指标与阈值无关。

{markdown_table(tables['reduction_comparison_default'], columns)}

'''
    if 'team_comparison' in tables:
        text += '\n## 小组测试结果\n\n' + markdown_table(tables['team_comparison'], columns)
        text += '\n\n全部使用各自验证集所选阈值。MLE/MAP 引用 B 经来源核对的保存结果，采用参考编码 41 列；C 的 NB 降维实验采用完整编码。\n'
    text += f'''
## 控制实验与误分类

中心化后的有效维数为 {summary['rank']['centered_covariance_eigenvalue_gt_1e-10']}，未中心化矩阵的数值秩为 {summary['rank']['uncentered_numpy_matrix_rank']}。
验证集平滑量控制见 smoothing_ablation_validation.csv，类内协方差统计见 coordinate_diagnostics_train.csv，重构误差见 metrics.json。
Gaussian NB 的对角方差假设依赖坐标系，完整方差保留与分类性能应分别评价。

misclassified_samples.csv 保存主阈值下的全部 FP/FN，source_record 为原始 CSV 从 1 开始、不含表头的记录号。
misclassification_summary.csv 和 misclassification_feature_summary.csv 提供计数及特征统计。
qualitative_cases.csv 按记录号选取各类预测转变的案例。完整分数与两套预测可通过返回值 tables['test_predictions'] 获取。

运行命令见 [README](../README.md)，完整指标见 metrics.json。本轮结果基于同一次数据划分。
'''
    if make_plots:
        text += '''
## 图表

![验证集 k 与训练方差](performance_vs_k.png)

![测试表示比较](method_comparison.png)

![解释方差](explained_variance.png)

![主成分系数](pca_loadings.png)

![二维投影](pc1_pc2.png)

其他图表见 test_pr_curves.png、smoothing_ablation.png。
'''
    (out/'RESULTS.md').write_text(text, encoding='utf-8')
