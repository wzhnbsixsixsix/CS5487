# Bank Marketing PCA 与 SVD 降维分析

bank_marketing_c.py 提供手写 PCA、SVD、不同主成分数量实验、分类性能比较和误分类分析，支持脚本运行与函数调用。使用原版 bank-full.csv，以 y=yes 为正类，默认排除 duration。

## 运行

将 bank+marketing.zip 放在仓库根目录，在本目录执行：

```powershell
python -m pip install -r requirements.txt
python bank_marketing_c.py "../bank+marketing.zip" --split-indices ../bank-marketing-a/results_a/split_indices.csv --out results_c
```

数据参数也可指向 bank.zip 或 bank-full.csv。预处理模块默认路径为 ../bank-marketing-a/bank_marketing_a.py，可用 --a-module 指定其他路径。

主要参数：--k 5 10 20 30 40 50 指定候选维数，--select-metric pr_auc 指定验证集选择指标，--seed 42 设置划分种子，--include-duration 加入通话时长，--out 指定输出目录。提供 --split-indices 时直接使用保存的划分。

## 方法

| 内容 | 实现 |
| --- | --- |
| PCA | 训练集中心化、协方差矩阵、NumPy 特征值分解与投影 |
| SVD | NumPy 经济型 SVD，截取前 k 个右奇异向量，不自动中心化 |
| k 扫描 | 验证集 PR-AUC 选择，并列时取较小 k |
| 分类比较 | 相同训练/测试记录与分类器配置，比较 Original、PCA、SVD |
| 评价 | Accuracy、Precision、Recall、F1、ROC-AUC、梯形 PR-AUC |
| 可视化 | 解释方差、主成分系数热图、二维投影、k 曲线、方法比较 |
| 误分类 | FP/FN 明细、计数和特征统计 |

预处理仅在训练集拟合：数值列标准化，独热列保留 0/1。默认分类器为 GaussianNB，var_smoothing=0.01，分类阈值为 log-odds ≥ 0；AUC 使用连续 log-odds。k 在验证集选择后固定，测试集用于最终比较。

## 函数调用

在本目录执行 python -m pip install -e . 后，可在其他目录导入：

```python
from cs5487_c_group import ManualPCA, ManualSVDReducer

pca = ManualPCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svd = ManualSVDReducer(n_components=4)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)
```

run_analysis(X_train, X_eval, y_train, y_eval, feature_names, ...) 接收已有数值数据，返回指标表、拟合模型和误分类统计。run_group_a_experiment(data_path, ...) 执行完整的三集实验。两者均支持 classifier_factory 参数替换分类器。

## 结果

[实验结果](results_c/RESULTS.md)包含验证集选择、测试集指标和图表。results_c/ 保存指标 JSON、比较表、k 扫描表、划分索引、误分类明细及五张 PNG。