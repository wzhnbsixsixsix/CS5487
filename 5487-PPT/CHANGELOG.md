# Changelog

## 2026-11-27 — Member B 两页图表修订

- 以小组成稿为底稿，从提交 `17339da` 的旧版 Beamer 稿恢复 Member B 的方法和核心结果。第 3 页的固定呼叫预算比较及其讲者备注原样保留，全稿仍为 8 页。
- 第 4 页新增逻辑回归训练目标及系数范数条形图，展示 MLE（λ = 0）和高斯先验 MAP（λ = 10）。系数范数由 3.771 降到 3.454，减少 8.4%。讲者备注记录交叉验证、截距处理和精确数值。
- 第 5 页以堆叠条形图展示各模型采用验证集 F1 阈值时的实际呼叫结果。MAP 多找到 35 位认购者，同时增加 128 通无效电话。保留共用 0.50 阈值时的对照，将精确 PR-AUC、分组误差、阈值和交接稿放入讲者备注。
- 更新 `README.md` 的页码说明。方法和结果依据 [Member B 实验结果](../bank-marketing-b/results_b/RESULTS.md)及其配套指标。检查了演示稿包内文件和逐页渲染，确认只有第 4、5 页及其讲者备注发生变化。

## 2026-11-27 — 小组最终演示稿

- 导入组员提供的 `CS5487_Bank_Marketing_ABCD_Final_Polished.pptx`，保留原文件内容。成稿共 8 页，第 3–5 页包含 Member B 的 MLE 与 MAP 逻辑回归比较。
- 更新 `README.md`，将打开演示稿的入口改为最终 PowerPoint 文件，并说明 Member B 内容所在页码。
- 移除旧版 Beamer 源文件 `main.tex`、主题 `beamerthemeCityUDG.sty`、校徽素材 `figures/`、构建输出 `build/`、旧预览 `preview/`、模板说明 `NOTICE.md` 和 Finder 目录元数据 `.DS_Store`。旧文件及其中的 Member B 两页可在提交 `17339da` 的 `5487-PPT/` 下查阅。
- 验证 PowerPoint 文件的压缩包完整性，并逐字节比对仓库副本与组员原稿。

## 2026-11-19 — Standalone template

- Extracted a standalone theme repository without the course-notes Git history or unrelated files.
- Retained the supplied Dongguan university logo and the burgundy/red/orange visual design; removed unused legacy color aliases.
- Replaced outdated FYP/2022 metadata and Hong Kong campus placeholders with editable CityUHK (Dongguan) bilingual examples.
- Added 16:9 examples covering headings, sections, lists, blocks, formulas and tables; pinned example CJK fonts to portable Fandol fonts.
- Removed local workspace files, VS Code user configuration scripts, machine-specific build entry points, old logos and unused course figures from the published file set.
- Rewrote README for portable XeLaTeX use; added source/status notice and ignored generated files.
- Validation results are recorded below after compilation and visual review.

### Validation

- Compiled with TeX Live 2026, XeLaTeX and latexmk; all 10 pages generated successfully with SyncTeX.
- Checked the final LaTeX log for overfull boxes and missing characters; none reported.
- Rendered all 10 pages with macOS PDFKit and reviewed Chinese/English text, school logo, title panel, section dividers, footer page counts, blocks, equations and tables; no clipping or overlapping text found.
- The bundled Poppler renderer lacked Adobe-GB1 mapping data, so its incomplete Chinese rendering was not used for the published previews; PDFKit confirmed Chinese text rendered correctly.
- Published source contains no local runtime paths or course-specific dependencies. Preview PDF and title image correspond to the included source.
