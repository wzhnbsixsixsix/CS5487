# Changelog

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
