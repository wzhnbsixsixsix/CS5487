# Bank Marketing · 成员 B

在成员 A 的固定划分上比较普通逻辑回归的最大似然估计（MLE）和零均值高斯先验下的最大后验估计（MAP）。同时重新运行 A 的高斯朴素贝叶斯，作为测试集对照。

## 复现

需要 Python 3.10 或更新版本。下载 [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing) 的完整压缩包，将它放到仓库根目录，并在该目录执行：

```bash
python -m pip install -r bank-marketing-b/requirements.txt
python bank-marketing-b/bank_marketing_b.py "bank+marketing.zip"
```

输入也可用压缩包内的 `bank.zip` 或原始 `bank-full.csv`。通过 `--out PATH` 可另存结果。脚本会检查全部行号与标签是否匹配 [A 的固定划分](../bank-marketing-a/results_a/split_indices.csv)。运行时版本和数据指纹见 [metrics.json](results_b/metrics.json)。指纹为逐行对 `[原始字段, 0/1 标签]` 的键排序紧凑 UTF-8 JSON（每条以 LF 结尾）计算 SHA-256。压缩格式与 CSV 换行差异不影响指纹。

数据由 Moro、Rita 和 Cortez 提供，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。[数据集 DOI](https://doi.org/10.24432/C5K306)

## 实验流程

1. 沿用 A 的随机种子 42、70%／15%／15% 分层划分及预处理，排除通话结束后才知道的 `duration`。训练、验证、测试分别有 31,648、6,782、6,781 条。
2. 在 A 的训练集内用种子 42 做分层 5 折。每折编码词表和标准化参数只从该折训练部分估计。两种逻辑回归都把每个分类型字段的第一个排序类别设为参考水平，最终使用同一组 41 列。A 的高斯 NB 使用原有完整独热编码。
3. 手写总负对数似然及梯度，使用 SciPy L-BFGS 求解。MAP 在总负对数似然上加 `λ × ||β||² / 2`，对应独立的 `N(0, 1/λ)` 系数先验，截距不受惩罚。MLE 取 λ = 0。按平均折内梯形 PR-AUC 在正 λ 中选择 MAP，并列取较小的 λ。
4. 最终模型在完整训练集上拟合。每个模型独立选择在验证集上 F1 最高的阈值，并列取较高阈值。固定阈值后评估测试集，同时记录 0.5 阈值的结果。ROC-AUC、梯形 PR-AUC 和 AP 使用连续分数。

## 结果文件

| 文件 | 内容 |
| --- | --- |
| [metrics.json](results_b/metrics.json) | 数据指纹、运行版本、λ 网格、参考水平、收敛信息、两套阈值的指标和混淆矩阵 |
| [cv_pr_auc.csv](results_b/cv_pr_auc.csv) / [cv_fold_indices.csv](results_b/cv_fold_indices.csv) | λ 扫描结果和每折验证部分的原始记录号 |
| [coefficients.csv](results_b/coefficients.csv) | MLE、MAP 系数及截距 |
| [error_groups.csv](results_b/error_groups.csv) / [misclassified_test.csv](results_b/misclassified_test.csv) | 验证集所选阈值下的分组统计及全部测试错例，`source_record` 从原始 CSV 第一条数据起计数 |
| [test_comparison.png](results_b/test_comparison.png) | 同一测试集上的 PR 曲线及误报、漏报计数 |
| [RESULTS.md](results_b/RESULTS.md) | 三模型测试比较、错例解读及给成员 C 的结论 |

## 成员 C 整合材料

- [报告 LaTeX 片段](handoff/report_b.tex)分别对应课程 LNCS 报告的 Method 和 Experiments and Results。将相关小节放入团队模板，复制结果图并按项目图片路径修改 `\includegraphics`。在参考文献中加入 UCI 数据集引用。
- [两页 PPT](handoff/member_b.pptx)包含方法、对照结果和演讲者备注，可复制到小组幻灯片。备注约 55–65 秒。
- 整合时保留测试集共 6,781 条（793 个正例）、梯形 PR-AUC 口径和各模型验证集所选阈值。若另用 0.5 阈值，表格需要注明。

## 修改记录

2026-11-22 新增 B 的训练、交叉验证和评估入口，复用 A 的拆分、编码与评估。新增可复现结果、误分类分析、报告片段和供 C 合并的演示页。实验文件由运行入口生成，`RESULTS.md` 和 `handoff/` 依据该次实验编写。
