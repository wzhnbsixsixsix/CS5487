# Bank Marketing · 成员 D — 类别不平衡的敏感性分析

`bank_marketing_d.py` 把 A、B、C 三人的五个模型放在**同一张测试集打分表**上，再做两个敏感性实验：**操作点/代价敏感性（S1）**与**正类先验漂移（S3）**。

本目录只写 `results_d/`，不修改 A/B/C 的任何文件；五个模型全部通过 import 他们的模块重新拟合。

## 复现

需要 A、B、C 三人的目录与根目录的 `bank+marketing.zip` 都在位，并在仓库根目录执行：

```powershell
env.cmd bank-marketing-d\bank_marketing_d.py "bank+marketing.zip" --stage all
```

`env.cmd` 是仓库根目录的启动器（使用根目录 `.venv`）。三个 stage 也可单独运行：

| stage | 作用 | 耗时 |
| --- | --- | --- |
| `scores` | 拟合五个模型，导出统一打分表并做复现校验 | 6 秒 |
| `s1` | 操作点与代价敏感性（只读 `test_scores.csv`，不重新训练） | 7 秒 |
| `s3` | 先验漂移敏感性（同上） | 7 秒 |

主要参数：`--out` 指定输出目录，`--lambda-map` 指定 MAP 的 λ（默认 10，B 选出的值），`--k` 指定 PCA/SVD 维数（默认 45，C 选出的值）。

## 为什么先造一张打分表

A 只保存了指标与错例，没有保存连续分数；B 和 C 同样只保存指标。**因此任何跨模型的阈值比较都无法在他们的产物上完成。**

`results_d/test_scores.csv` 把五个模型在 6,781 条测试记录上的 log-odds 并列保存：

```
source_record,y_true,gnb,lr_mle,lr_map,pca_gnb,svd_gnb
```

之后所有实验只读这一张表，不再训练。表中每个模型重算出的 ROC-AUC 与 PR-AUC 都要先与 A/B/C 报告值核对，误差小于 5×10⁻⁷ 才继续。

## 方法

| 实验 | 做法 |
| --- | --- |
| **S1 操作点/代价敏感性** | 固定打分表，把决策阈值从 0 扫到 1。定义总代价 `FP + r·FN`（`r` 为漏报代价与误报代价之比），对每个 `r` 在 1–20 之间取各模型的最优阈值，比较四类对象：五个模型、团队现有阈值、全打、全不打 |
| **S3 先验漂移敏感性** | 给测试集加权（正类 `π′/π`，负类 `(1−π′)/(1−π)`），把正类比例人为改到 2%–60% 共 60 个档位，**不增删、不重标任何记录**，重算 ROC-AUC、梯形 PR-AUC、相对基线倍数与 Brier 分数 |

## 结果文件

| 文件 | 内容 |
| --- | --- |
| [metrics.json](results_d/metrics.json) | 五个模型的重算指标、验证集所选阈值、与 A/B/C 的逐项复现校验 |
| `test_scores.csv` / `validation_scores.csv` | 统一打分表（log-odds） |
| [s1_operating_points.csv](results_d/s1_operating_points.csv) | 团队规则下每个模型的阈值、判正人数与混淆矩阵 |
| [s1_flip_table.csv](results_d/s1_flip_table.csv) | 8 个代价比下的最优模型、团队方案的超额支出、与"全不打/全打"基线的对比 |
| `s1_cost_curves.csv` | 77 个代价比 × 5 模型的完整代价曲线 |
| [s3_metrics.csv](results_d/s3_metrics.csv) | 60 个正类比例 × 5 模型的 ROC-AUC、PR-AUC、倍数、Brier |
| [s3_thresholds.csv](results_d/s3_thresholds.csv) | 每个正类比例下各模型的 F1 最优阈值 |
| [RESULTS.md](results_d/RESULTS.md) | 结论、表格与两张图 |
| `s1_f1_vs_threshold.png`、`s1_cost_vs_ratio.png`、`s3_prior_shift.png` | 图（同时提供 300 dpi 的 PDF） |

## 与 A/B/C 的口径对齐

- 梯形 PR-AUC，不使用 AP；与 B 在 README 中的声明一致。
- 阈值一律取**验证集上 F1 最大**的那一个，并列时取较高阈值（沿用 A 的 `select_threshold`）。
- 复跑 B 的 λ 选择按固定值 10 写入，不重新做 5 折交叉验证；该值已由 B 报告并通过复现校验。
- ⚠️ **与 C 的口径差异**：C 的测试集指标使用固定阈值 `log-odds ≥ 0`，本目录统一改用验证集所选阈值。因此 PCA/SVD 的 F1 由 0.3780/0.3759 变为 0.3855/0.3769；排序指标（ROC-AUC、PR-AUC）完全一致。

## 给 C 的整合材料

- [报告 LaTeX 片段](handoff/report_d.tex)：对应 LNCS 报告的 Method 与 Experiments and Results，约 1 页。含一张表与两张图，其中 S3 图标注为"版面紧张时可删"。
- 两张图已在 `results_d/` 中，复制时按项目图片路径修改 `\includegraphics`。

整合时请保留三条口径：测试集共 6,781 条（793 个正例）、梯形 PR-AUC、以及"PR-AUC 数值依赖于 11.69% 的正类比例"这一声明。

## 修改记录

2026-11-23 新增统一打分表、S1 与 S3 两个敏感性实验、复现校验与报告片段。实验文件由运行入口生成，`RESULTS.md` 与 `handoff/` 依据该次实验编写。
