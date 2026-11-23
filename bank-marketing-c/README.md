# Bank Marketing PCA 与 SVD 降维分析

提供手写 PCA、SVD、不同 k 的分类性能比较、可视化和误分类分析，支持脚本运行与函数调用。

## 运行

需要 Python 3.10+。将 bank+marketing.zip 放在仓库根目录，在本目录执行：

```powershell
python -m pip install -r requirements.txt
python bank_marketing_c.py "../bank+marketing.zip" --split-indices ../bank-marketing-a/results_a/split_indices.csv --b-metrics ../bank-marketing-b/results_b/metrics.json --out results_c
```

数据参数也可指向 bank.zip 或 bank-full.csv。主要参数：`--k 1 5 10 20 40 41 42 45 50` 指定候选维数，`--select-metric pr_auc` 指定验证指标，`--no-plots` 仅生成数值结果，`--out` 指定输出目录。省略 `--b-metrics` 时仅生成 C 的比较结果。

## 方法

| 内容 | 实现 |
| --- | --- |
| 数据 | 复用 A 的划分与预处理，默认排除 duration；数值标准化、完整独热编码共 50 列 |
| PCA | 训练集中心化、协方差矩阵、NumPy 特征值分解 |
| SVD | 未中心化矩阵的 NumPy 经济型 SVD，保留前 k 个方向 |
| k 与阈值 | PCA/SVD 分别按验证 PR-AUC 选 k，再按验证 F1 选分类阈值 |
| 分类与评价 | A 手写 Gaussian NB；Accuracy、Precision、Recall、F1、ROC-AUC、梯形 PR-AUC 与 AP |
| 分析与图表 | 降维前后、固定 k、平滑量控制、误分类统计及七类 PNG 图表 |

所有拟合仅用训练集，k 和阈值固定后评价测试集。小组比较表的 MLE/MAP 引用 B 经数据、划分和基线核对的保存指标。

## 函数调用

执行 `python -m pip install -e .` 后可在其他目录导入：

```python
from cs5487_c_group import ManualPCA, ManualSVDReducer, run_group_a_experiment

pca = ManualPCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svd = ManualSVDReducer(n_components=4)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)
```

`run_group_a_experiment(data_path, split_path=..., b_metrics_path=..., output_dir=...)` 执行完整实验，返回指标表、拟合模型与数据。`run_analysis(...)` 接收已有数值矩阵，支持 classifier_factory，自带默认分类器为 sklearn LogisticRegression。

## 结果

[实验结果](results_c/RESULTS.md)包含验证选择、测试比较、误分类统计和图表。默认数据为 45,211 条，训练/验证/测试分别为 31,648 / 6,782 / 6,781；PCA 选 41 维，SVD 选 42 维。精确配置与指标见 results_c/metrics.json，主表使用各表示自己的验证集阈值，另存零阈值辅助表。
