# Bank Marketing — 手写高斯朴素贝叶斯分类

`bank_marketing_a.py` 是可单独运行、也可供 B/C 同学导入的 Python 模块。默认读取 UCI 原版完整数据 `bank-full.csv`（45,211 条），以认购定期存款 `y=yes` 为正类。

## 数据来源

从 [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) 下载数据，将下载的 `bank+marketing.zip` 放在项目根目录。也可使用其中的 `bank.zip` 或 `bank-full.csv`。本项目使用原版 16 个输入字段的数据，目标列为 `y`。

数据由 Moro、Rita 和 Cortez 提供，数据许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

## 运行

把源码、`requirements.txt` 和课程数据 `bank+marketing.zip` 放在同一目录，从该目录执行：

```powershell
python -m pip install -r requirements.txt
python bank_marketing_a.py "bank+marketing.zip" --out results_a
```

数据参数也可直接指向下载目录的 ZIP、内层 `bank.zip`，或解压后的 `bank-full.csv`。脚本直接读 ZIP，无需先解压。默认随机种子是 42，可由 `--seed` 修改。默认排除 `duration`（最后一次通话时长，结束后才知道）；如果课程需要比较通话后的模型，用 `--include-duration --out results_with_duration` 单独运行。

## A 同学各项任务在代码中的位置

| 任务 | 代码 |
|---|---|
| 分层抽样和数据集划分 | `stratified_split`：分别打乱正类和负类索引，按 70%／15%／15% 分配余数 |
| 独热编码 | `fit_one_hot`、`one_hot_encode`：从训练集提取类别词表，再手工将对应位置置 1 |
| 标准化 | `fit_standardizer`、`standardize`：用训练集的均值和总体标准差做 Z-score，仅处理数值列 |
| 高斯朴素贝叶斯 | `GaussianNB`：自己估计先验、类内均值与方差、计算高斯对数似然与正负类对数分数差 |
| 混淆矩阵、准确率、精确率、召回率、F1 | `confusion_matrix`、`classification_metrics`：先数 TN、FP、FN、TP，再按定义计算 |
| ROC、PR、两种 AUC | `ranking_curves`、`trapezoid_area`：排序后按同分组累积 TP 和 FP，用梯形积分；同时给出按召回增量加权的 AP |
| 阈值扫描 | `select_threshold`：扫描验证集所有不同的 log-odds，选 F1 最高的阈值；并列时取较高阈值 |
| 误分类和图表 | `group_errors`、`collect_errors`、`plot_results`：按组计算误报率和漏报率，记录错误样本及主要字段贡献 |

NumPy 用于数组、随机打乱、排序和基础数学运算；Matplotlib 负责画图。项目代码不依赖 scikit-learn。高斯 NB 在每个类内方差上加一个小正数，默认是训练集最大列方差的 0.01 倍，用来平滑稀有独热列。类别字段的原始 `unknown` 保留；验证或测试出现训练集未见类别时，该字段独热列全为 0。`pdays=-1` 保留为数据定义的“此前未联系”标记。

## 和 B/C 同学共用数据切分

```python
from bank_marketing_a import load_bank_data, stratified_split, BankPreprocessor

rows, y = load_bank_data("bank+marketing.zip")
train, val, test = stratified_split(y, seed=42)

prep = BankPreprocessor().fit([rows[i] for i in train])
X_train, X_val, X_test = [
    prep.transform([rows[i] for i in indices]) for indices in (train, val, test)
]
y_train, y_val, y_test = y[train], y[val], y[test]
feature_names = prep.feature_names
```

跨语言复用时可直接读 `results_a/split_indices.csv`；`source_record` 为原始 CSV 的第几条数据（从 1 开始，不含表头），对应 Python 行索引 `source_record - 1`。团队比较模型时，应共用测试记录和是否使用 `duration` 的设定。

## 实验结果

仓库中的 `results_a/` 保存了随机种子 42、排除 `duration` 的运行结果。测试集 ROC-AUC 为 0.7497，梯形 PR-AUC 为 0.3592。完整指标与局限见该目录的 `RESULTS.md`。

## 输出

- [RESULTS.md](results_a/RESULTS.md)：实际运行的指标、误分类分析和图表。
- `test_evaluation.png`：ROC、PR 和两套阈值的混淆矩阵。
- `validation_thresholds.png/.csv`：验证集阈值扫描，CSV 的 `threshold` 为 log-odds。
- `test_error_groups.png`、`error_groups.csv`：按联系渠道、上次营销结果及职业分析误报与漏报。
- `misclassified_test.csv`：使用验证集所选阈值时的全部测试集误分类记录。
- `metrics.json`、`split_indices.csv`：数值结果、特征顺序和样本划分。

数据引用：Moro, S., Rita, P., & Cortez, P. (2014). [Bank Marketing](https://doi.org/10.24432/C5K306). UCI Machine Learning Repository.
