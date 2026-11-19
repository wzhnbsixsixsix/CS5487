"""Bank Marketing：预处理、手写 Gaussian NB、评估与误分类分析。"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Sequence
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np


NUMERIC_COLUMNS = ("age", "balance", "day", "campaign", "pdays", "previous")
CATEGORICAL_COLUMNS = (
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
)
SPLIT_NAMES = ("train", "validation", "test")
RATIOS = (0.70, 0.15, 0.15)
Row = dict[str, str]


def load_bank_data(path: str | Path) -> tuple[list[Row], np.ndarray]:
    """读取 UCI 外层 ZIP、内层 bank.zip、或原版 bank 系列分号 CSV。"""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            if "bank-full.csv" in archive.namelist():
                raw = archive.read("bank-full.csv")
            elif "bank.zip" in archive.namelist():
                with ZipFile(io.BytesIO(archive.read("bank.zip"))) as inner:
                    raw = inner.read("bank-full.csv")
            else:
                raise ValueError("压缩包中没有 bank.zip 或 bank-full.csv。")
    else:
        raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";")
    required = set((*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, "duration", "y"))
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"列不匹配：缺少 {', '.join(sorted(missing))}；请使用 bank-full.csv。")
    rows, labels = [], []
    for number, row in enumerate(reader, start=1):
        if None in row or any(not row[key] for key in required):
            raise ValueError(f"第 {number} 条数据记录存在空字段或列数不符。")
        label = row.pop("y")
        if label not in {"yes", "no"}:
            raise ValueError(f"第 {number} 条数据记录的 y 不是 yes/no。")
        labels.append(int(label == "yes"))
        rows.append(row)
    if not rows:
        raise ValueError("数据集没有记录。")
    return rows, np.asarray(labels, dtype=np.int8)


def stratified_split(
    y: np.ndarray, ratios: Sequence[float] = RATIOS, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分别打乱各类索引并分配；用最大余数法处理非整数样本数。"""
    y = np.asarray(y)
    ratios = np.asarray(ratios, dtype=float)
    if ratios.shape != (3,) or not np.all(ratios > 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("三个划分比例必须为正且总和为 1。")
    if y.ndim != 1 or set(y) != {0, 1}:
        raise ValueError("y 必须是一维二分类标签，且同时含 0 和 1。")
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        exact = len(indices) * ratios / ratios.sum()
        counts = np.floor(exact).astype(int)
        leftovers = len(indices) - int(counts.sum())
        order = sorted(range(3), key=lambda k: (-(exact[k] - counts[k]), k))
        for k in order[:leftovers]:
            counts[k] += 1
        if np.any(counts == 0):
            raise ValueError("有类别太少，无法同时分到三个集合。")
        start = 0
        for k, count in enumerate(counts):
            parts[k].append(indices[start:start + count])
            start += count
    train, validation, test = (np.concatenate(part) for part in parts)
    for part in (train, validation, test):
        rng.shuffle(part)
    return train, validation, test


def fit_one_hot(rows: Sequence[Row], columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """训练集出现过的取值组成词表；unknown 保留为原始类别。"""
    return {column: tuple(sorted({row[column] for row in rows})) for column in columns}


def one_hot_encode(rows: Sequence[Row], levels: dict[str, tuple[str, ...]]) -> np.ndarray:
    """逐列查表置 1；拟合时没见过的类别在该字段的所有列上置 0。"""
    matrix = np.zeros((len(rows), sum(map(len, levels.values()))), dtype=float)
    offset = 0
    for column, values in levels.items():
        lookup = {value: offset + i for i, value in enumerate(values)}
        for i, row in enumerate(rows):
            position = lookup.get(row[column])
            if position is not None:
                matrix[i, position] = 1.0
        offset += len(values)
    return matrix


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """手写逐列总体均值和标准差；常量列使用 1 作为缩放因子。"""
    mean = x.sum(axis=0) / len(x)
    variance = ((x - mean) ** 2).sum(axis=0) / len(x)
    scale = np.sqrt(variance)
    return mean, np.where(scale > 0, scale, 1.0)


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


class BankPreprocessor:
    """在训练集 fit，之后各集合共用同一特征顺序、词表及缩放参数。"""

    def __init__(self, include_duration: bool = False):
        self.numeric_columns = NUMERIC_COLUMNS + (("duration",) if include_duration else ())
        self.levels = None

    def _numeric(self, rows: Sequence[Row]) -> np.ndarray:
        x = np.asarray([[float(row[name]) for name in self.numeric_columns] for row in rows])
        x = x.reshape(len(rows), len(self.numeric_columns))
        if not np.isfinite(x).all():
            raise ValueError("数值字段含 NaN 或无穷大。")
        return x

    def fit(self, rows: Sequence[Row]) -> BankPreprocessor:
        if not len(rows):
            raise ValueError("不能用空训练集拟合预处理。")
        self.mean, self.scale = fit_standardizer(self._numeric(rows))
        self.levels = fit_one_hot(rows, CATEGORICAL_COLUMNS)
        self.feature_names = list(self.numeric_columns) + [
            f"{column}={value}" for column, values in self.levels.items() for value in values
        ]
        return self

    def transform(self, rows: Sequence[Row]) -> np.ndarray:
        if self.levels is None:
            raise RuntimeError("请先对训练集调用 fit。")
        nums = standardize(self._numeric(rows), self.mean, self.scale)
        return np.column_stack((nums, one_hot_encode(rows, self.levels)))


class GaussianNB:
    """逐类估计先验、均值与方差，再在对数空间累加高斯密度。"""

    def __init__(self, var_smoothing: float = 0.01):
        if not np.isfinite(var_smoothing) or var_smoothing <= 0:
            raise ValueError("var_smoothing 必须为有限正数。")
        self.var_smoothing = var_smoothing
        self.mean = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianNB:
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        if x.ndim != 2 or not x.shape[1] or y.shape != (len(x),) or set(y) != {0, 1}:
            raise ValueError("X 须为二维矩阵，y 须与其行数匹配且同时包含 0 和 1。")
        if not np.isfinite(x).all():
            raise ValueError("训练特征必须为有限数值。")
        center = x.sum(axis=0) / len(x)
        pooled_var = ((x - center) ** 2).sum(axis=0) / len(x)
        self.epsilon = self.var_smoothing * max(float(pooled_var.max()), 1e-12)
        means, variances, priors = [], [], []
        for label in (0, 1):
            subset = x[y == label]
            mean = subset.sum(axis=0) / len(subset)
            variance = ((subset - mean) ** 2).sum(axis=0) / len(subset)
            means.append(mean)
            variances.append(variance + self.epsilon)
            priors.append(len(subset) / len(x))
        self.mean = np.asarray(means)
        self.var = np.asarray(variances)
        self.prior = np.asarray(priors)
        return self

    def _log_density(self, x: np.ndarray, k: int) -> np.ndarray:
        return -0.5 * (np.log(2 * np.pi * self.var[k]) + (x - self.mean[k]) ** 2 / self.var[k])

    def _checked_x(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("请先训练模型。")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[1]:
            raise ValueError("输入特征维数与训练特征不一致。")
        return x

    def log_joint(self, x: np.ndarray) -> np.ndarray:
        x = self._checked_x(x)
        return np.column_stack([
            np.log(self.prior[k]) + self._log_density(x, k).sum(axis=1)
            for k in (0, 1)
        ])

    def feature_log_odds(self, x: np.ndarray) -> np.ndarray:
        """每个编码列对 log-odds 的加性贡献；总分还包含先验 log-odds。"""
        x = self._checked_x(x)
        return self._log_density(x, 1) - self._log_density(x, 0)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        log_joint = self.log_joint(x)
        return log_joint[:, 1] - log_joint[:, 0]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """两列依次对应 no、yes；分别计算以尽量避免 1 - p 的精度损失。"""
        score = self.decision_function(x)
        return np.column_stack((sigmoid(-score), sigmoid(score)))

    def predict(self, x: np.ndarray, log_odds_threshold: float = 0.0) -> np.ndarray:
        return (self.decision_function(x) >= log_odds_threshold).astype(np.int8)


def sigmoid(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    z = np.exp(-np.abs(score))
    return np.where(score >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def confusion_matrix(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """行是真实标签，列是预测标签，顺序均为 [0, 1]。"""
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or y.shape != pred.shape or not set(y) <= {0, 1} or not set(pred) <= {0, 1}:
        raise ValueError("真实和预测标签须为等长的一维 0/1 序列。")
    return np.array([
        [np.count_nonzero((y == actual) & (pred == predicted)) for predicted in (0, 1)]
        for actual in (0, 1)
    ])


def divide_or_zero(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num, den = np.broadcast_arrays(num, den)
    return np.divide(num, den, out=np.zeros(num.shape, dtype=float), where=den != 0)


def classification_metrics(matrix: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {
        "accuracy": float(divide_or_zero(tn + tp, tn + fp + fn + tp)),
        "precision": float(divide_or_zero(tp, tp + fp)),
        "recall": float(divide_or_zero(tp, tp + fn)),
        "f1": float(divide_or_zero(2 * tp, 2 * tp + fp + fn)),
    }


def ranking_curves(y: np.ndarray, scores: np.ndarray) -> dict:
    """由高到低逐组扫阈值；同分记录一次加入，首点代表全判负。"""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != scores.shape or set(y) != {0, 1}:
        raise ValueError("曲线要求等长的一维标签和分数，且正负类都存在。")
    if not np.isfinite(scores).all():
        raise ValueError("曲线分数必须为有限数值。")
    order = np.argsort(-scores, kind="stable")
    s, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.r_[0, np.cumsum(labels, dtype=int)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    positives, negatives = int(tp[-1]), int(fp[-1])
    fn = positives - tp
    recall = tp / positives
    fpr = fp / negatives
    precision = divide_or_zero(tp, tp + fp)
    precision[0] = 1.0  # PR 曲线起点的绘图约定，非单阈值的 precision 定义。
    return {
        "threshold": np.r_[np.inf, s[ends]],
        "tp": tp, "fp": fp, "fn": fn, "tn": negatives - fp,
        "fpr": fpr, "recall": recall, "precision": precision,
        "accuracy": (tp + negatives - fp) / (positives + negatives),
        "f1": divide_or_zero(2 * tp, 2 * tp + fp + fn),
        "roc_auc": trapezoid_area(fpr, recall),
        "pr_auc": trapezoid_area(recall, precision),
        "average_precision": float((np.diff(recall) * precision[1:]).sum()),
    }


def trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    return float((np.diff(x) * (y[1:] + y[:-1]) * 0.5).sum())


def select_threshold(validation: dict) -> float:
    """验证集 F1 最大；若并列，取较高阈值以减少预测为正的人数。"""
    return float(validation["threshold"][1 + np.argmax(validation["f1"][1:])])


def group_errors(rows: Sequence[Row], y: np.ndarray, pred: np.ndarray) -> list[dict]:
    """某组 FPR = FP/真实负例；FNR = FN/真实正例。无分母时返回 None。"""
    output = []
    for feature in ("job", "contact", "poutcome"):
        buckets = {}
        for i, row in enumerate(rows):
            buckets.setdefault(row[feature], []).append(i)
        for value, indices in sorted(buckets.items()):
            tn, fp, fn, tp = map(int, confusion_matrix(y[indices], pred[indices]).ravel())
            n0, n1 = tn + fp, tp + fn
            output.append(dict(feature=feature, value=value, n=n0 + n1, n_no=n0, n_yes=n1,
                               tn=tn, fp=fp, fn=fn, tp=tp,
                               fpr=fp / n0 if n0 else None, fnr=fn / n1 if n1 else None))
    return output


def collect_errors(rows, source_indices, y, pred, scores, columns, feature_scores) -> list[dict]:
    """列出测试集的全部错误，按离决策线的方向查看高分 FP 与低分 FN。"""
    error_rows = []
    # 一个分类字段可能有多个独热列。先按原始字段相加，再找主要推分项。
    columns = [name.split("=", 1)[0] for name in columns]
    fields = list(dict.fromkeys(columns))
    contributions = np.column_stack([
        feature_scores[:, [i for i, column in enumerate(columns) if column == field]].sum(axis=1)
        for field in fields
    ])
    for kind, actual in (("FP", 0), ("FN", 1)):
        indices = np.flatnonzero((y == actual) & (y != pred))
        direction = 1 if kind == "FP" else -1
        order = np.argsort(-direction * scores[indices], kind="stable")
        for i in indices[order]:
            largest = int(np.argmax(direction * contributions[i]))
            field = fields[largest]
            error_rows.append(dict(
                error=kind, source_record=int(source_indices[i]) + 1,
                actual=int(y[i]), predicted=int(pred[i]), log_odds=float(scores[i]),
                top_feature=field, top_value=rows[i][field],
                top_log_odds_contribution=float(contributions[i, largest]),
                **rows[i],
            ))
    return error_rows


def write_csv(path: Path, columns: Sequence[str], records) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def plot_results(out, test, validation, threshold, matrices, groups, prevalence) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    navy, teal, amber, gray = "#18354d", "#067874", "#b6522a", "#64748b"
    plt.rcParams.update({
        "font.size": 11, "text.color": navy, "axes.labelcolor": navy,
        "axes.titlecolor": navy, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#d8e1e8", "axes.axisbelow": True, "savefig.dpi": 160,
    })
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.suptitle("Bank Marketing  |  Gaussian Naive Bayes on held-out test data",
                 fontsize=17, fontweight="bold")
    roc, pr, default_cm, tuned_cm = axs.flat
    roc.plot(test["fpr"], test["recall"], color=teal, lw=2.2,
             label=f'ROC-AUC {test["roc_auc"]:.3f}')
    roc.plot([0, 1], [0, 1], color=gray, ls="--", lw=1, label="Chance")
    roc.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    pr.plot(test["recall"], test["precision"], color=teal, lw=2.2,
            label=f'PR-AUC {test["pr_auc"]:.3f}  /  AP {test["average_precision"]:.3f}')
    pr.axhline(prevalence, color=gray, ls="--", lw=1, label=f"Positive share {prevalence:.3f}")
    pr.set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    for ax in (roc, pr):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.7)
        ax.legend(loc="best", frameon=False, fontsize=9)
    maximum = max(int(m.max()) for m in matrices)
    for ax, m, title in zip((default_cm, tuned_cm), matrices,
                            ("Default threshold", "Threshold selected on validation")):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=maximum)
        ax.set(title=title, xlabel="Predicted", ylabel="Actual",
               xticks=[0, 1], xticklabels=["no", "yes"], yticks=[0, 1], yticklabels=["no", "yes"])
        for iy in range(2):
            for ix in range(2):
                ax.text(ix, iy, f"{m[iy, ix]:,}", ha="center", va="center", fontweight="bold",
                        color="white" if m[iy, ix] > maximum * 0.55 else navy, fontsize=14)
    fig.savefig(out / "test_evaluation.png", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    log_odds = validation["threshold"][1:]
    ax.plot(log_odds, validation["f1"][1:], color=teal, label="F1", lw=2)
    ax.plot(log_odds, validation["recall"][1:], color=navy, label="Recall", alpha=0.75)
    ax.plot(log_odds, validation["precision"][1:], color=amber, label="Precision", alpha=0.75)
    ax.axvline(0, color=gray, ls=":", lw=1.2, label="Default")
    ax.axvline(threshold, color=teal, ls="--", lw=1.2, label=f"Selected ({threshold:.2f})")
    # 对称对数轴同时保留远端极值和接近决策线的阈值。
    ax.set_xscale("symlog", linthresh=10)
    ax.set(title="Threshold scan on validation data", xlabel="Log-odds threshold (symlog scale)",
           ylabel="Metric", ylim=(0, 1.03))
    ax.grid(alpha=0.7)
    ax.legend(loc="best", frameon=False, ncols=2)
    fig.savefig(out / "validation_thresholds.png", facecolor="white")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="constrained")
    relevant = [g for g in groups if g["feature"] in {"contact", "poutcome"}]
    positions = np.arange(len(relevant))
    fig.suptitle("Where test errors occur  |  Validation-selected threshold",
                 fontsize=16, fontweight="bold")
    for ax, rate, denom, count, color, title in (
        (axs[0], "fpr", "n_no", "fp", amber, "False positives among actual no"),
        (axs[1], "fnr", "n_yes", "fn", teal, "False negatives among actual yes"),
    ):
        for i, g in enumerate(relevant):
            r = g[rate]
            if r is not None:
                ax.barh(i, 100 * r, color=color, height=0.55)
            label = f'{g[count]}/{g[denom]}' if r is not None else "n/a"
            ax.text(100 * (r or 0) + 2, i, label, ha="left", va="center", fontsize=9)
        ax.set(title=title, xlim=(0, 122), xticks=[0, 25, 50, 75, 100], xlabel="Error rate (%)")
        ax.grid(axis="x", alpha=0.7)
    axs[0].set_yticks(positions, [f'{g["feature"]}: {g["value"]}' for g in relevant])
    axs[0].invert_yaxis()
    fig.savefig(out / "test_error_groups.png", facecolor="white")
    plt.close(fig)


def write_report(path, summary, groups, errors, sample_name) -> None:
    splits = summary["splits"]
    base, tuned = summary["test"]["default"], summary["test"]["validation_selected"]
    curve = summary["test"]["ranking"]
    split_lines = [
        f'| {name} | {s["n"]:,} | {s["no"]:,} | {s["yes"]:,} | {s["yes_rate"]:.2%} |'
        for name, s in splits.items()
    ]
    columns = ("accuracy", "precision", "recall", "f1")
    metric_lines = [
        f'| {name} | ' + " | ".join(f'{m[key]:.4f}' for key in columns) + ' |'
        for name, m in (("默认阈值", base), ("验证集所选阈值", tuned))
    ]
    tn0, fp0, fn0, tp0 = (n for row in base["confusion_matrix"] for n in row)
    tn1, fp1, fn1, tp1 = (n for row in tuned["confusion_matrix"] for n in row)
    group_lines = []
    for feature in ("contact", "poutcome", "job"):
        for r, n, c, term in (("fpr", "n_no", "fp", "误报"), ("fnr", "n_yes", "fn", "漏报")):
            candidates = [g for g in groups if g["feature"] == feature and g[n] >= 50]
            if candidates:
                g = max(candidates, key=lambda row: row[r])
                group_lines.append(
                    f'- `{feature}={g["value"]}`：在该字段分母至少 50 的组中{term}率最高，'
                    f'为 {g[c]:,}/{g[n]:,}（{g[r]:.1%}）。'
                )
    example_lines = []
    for kind in ("FP", "FN"):
        for e in [entry for entry in errors if entry["error"] == kind][:2]:
            example_lines.append(
                f'| {kind} | {e["source_record"]:,} | {e["job"]} | {e["contact"]} | '
                f'{e["poutcome"]} | {e["log_odds"]:.2f} | '
                f'`{e["top_feature"]}={e["top_value"]}` ({e["top_log_odds_contribution"]:+.2f}) |'
            )
    text = "\n".join([
        "# Bank Marketing：A 实验结果", "",
        f"输入：`{sample_name}`；共有 {sum(s['n'] for s in splits.values()):,} 条记录。"
        f"正类为认购定期存款（`y=yes`）。随机种子 {summary['seed']}。"
        f"使用 {summary['n_features']} 个编码后的特征，"
        f"`duration`{'已纳入本次对照实验' if summary['include_duration'] else '未用于训练'}。", "",
        "| 划分 | 样本数 | no | yes | 正类占比 |", "|---|---:|---:|---:|---:|",
        *split_lines, "",
        "## 测试集表现", "",
        "| 决策规则 | 准确率 | 精确率 | 召回率 | F1 |", "|---|---:|---:|---:|---:|",
        *metric_lines, "",
        f'ROC-AUC **{curve["roc_auc"]:.4f}**，梯形 PR-AUC **{curve["pr_auc"]:.4f}**，'
        f'AP **{curve["average_precision"]:.4f}**。PR 曲线的随机参考线为测试集正类占比 '
        f'**{splits["test"]["yes_rate"]:.2%}**。始终预测 no 的准确率为 '
        f'{1 - splits["test"]["yes_rate"]:.2%}，正类召回率为 0。'
        "因此同时看精确率、召回率和 PR 曲线。", "",
        "默认规则用 log-odds ≥ 0（模型计算的正类后验 ≥ 0.5）。"
        f"验证集所选阈值为 **{tuned['log_odds_threshold']:.4f}**，"
        f"该阈值在验证集的 F1 为 **{summary['validation_best_f1']:.4f}**。"
        "ROC-AUC 和 PR-AUC 使用同一组连续分数，因此两条决策规则共用曲线。", "",
        f"默认阈值：TN {tn0:,}、FP {fp0:,}、FN {fn0:,}、TP {tp0:,}。"
        f"验证集阈值：TN {tn1:,}、FP {fp1:,}、FN {fn1:,}、TP {tp1:,}。"
        f"应用验证集阈值后，误报变化 {fp1 - fp0:+,} 条，漏报变化 {fn1 - fn0:+,} 条，"
        f"F1 变化 {tuned['f1'] - base['f1']:+.4f}。", "",
        "![测试集 ROC、PR 曲线及混淆矩阵](test_evaluation.png)", "",
        "![验证集阈值扫描](validation_thresholds.png)", "",
        "## 误分类样本", "",
        "以下均为验证集所选阈值在测试集上的表现。误报是把未认购预测为认购；漏报是错过认购者。",
        "误报率在每组真实 no 中计算，漏报率在每组真实 yes 中计算。", "",
        *group_lines, "",
        "![测试集分组误报和漏报](test_error_groups.png)", "",
        "下面列出分数最高的误报、分数最低的漏报。原始记录号从第一条数据开始计数（不含表头）。"
        "各项贡献来自两类的对数似然之差，同一字段的独热列合并；误报取最多推向 yes 的字段，"
        "漏报取最多推向 no 的字段。", "",
        "| 类型 | 原始记录号 | 职业 | 联系方式 | 上次营销结果 | log-odds | 主要字段贡献 |",
        "|---|---:|---|---|---|---:|---|", *example_lines, "",
        "完整错误记录见 `misclassified_test.csv`；完整分组统计见 `error_groups.csv`。"
        "组间差异是本次切分下的描述性统计，无法从中推出因果关系。"
        "高斯 NB 将数值列和独热列统一建模，并假设给定类别后各列独立；"
        "同一字段的独热列实际互斥，模型分数不宜当作校准后的认购概率。"
        "该实验使用随机分层切分，未验证未来时段上的表现。", "",
        "数据来源：[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)。", "",
    ])
    path.write_text(text, encoding="utf-8")


def run(path: Path, out: Path, seed: int = 42, include_duration: bool = False,
        var_smoothing: float = 0.01) -> dict:
    rows, y = load_bank_data(path)
    train, val, test = stratified_split(y, seed=seed)
    train_rows, val_rows, test_rows = ([rows[i] for i in ids] for ids in (train, val, test))
    preprocessor = BankPreprocessor(include_duration).fit(train_rows)
    x_train, x_val, x_test = (preprocessor.transform(r) for r in (train_rows, val_rows, test_rows))
    model = GaussianNB(var_smoothing).fit(x_train, y[train])
    validation = ranking_curves(y[val], model.decision_function(x_val))
    threshold = select_threshold(validation)
    scores = model.decision_function(x_test)
    curves = ranking_curves(y[test], scores)
    pred_default = (scores >= 0).astype(np.int8)
    pred_selected = (scores >= threshold).astype(np.int8)
    matrices = [confusion_matrix(y[test], pred) for pred in (pred_default, pred_selected)]
    groups = group_errors(test_rows, y[test], pred_selected)
    errors = collect_errors(test_rows, test, y[test], pred_selected, scores,
                            preprocessor.feature_names, model.feature_log_odds(x_test))

    summary = dict(
        seed=seed, ratios=list(RATIOS), include_duration=include_duration,
        var_smoothing=var_smoothing, n_features=len(preprocessor.feature_names),
        feature_names=preprocessor.feature_names, validation_best_f1=float(validation["f1"].max()),
        splits={name: dict(n=len(ids), no=int((y[ids] == 0).sum()),
                           yes=int(y[ids].sum()), yes_rate=float(y[ids].sum() / len(ids)))
                for name, ids in zip(SPLIT_NAMES, (train, val, test))},
        test=dict(
            default=dict(log_odds_threshold=0.0, confusion_matrix=matrices[0].tolist(),
                         **classification_metrics(matrices[0])),
            validation_selected=dict(log_odds_threshold=threshold, confusion_matrix=matrices[1].tolist(),
                                     **classification_metrics(matrices[1])),
            ranking={m: curves[m] for m in ("roc_auc", "pr_auc", "average_precision")},
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out / "split_indices.csv", ("split", "source_record", "y"), (
        dict(split=name, source_record=int(i) + 1, y=int(y[i]))
        for name, ids in zip(SPLIT_NAMES, (train, val, test)) for i in ids
    ))
    threshold_cols = ("threshold", "tn", "fp", "fn", "tp", "accuracy", "precision", "recall", "f1")
    write_csv(out / "validation_thresholds.csv", threshold_cols, (
        {key: validation[key][i] for key in threshold_cols}
        for i in range(1, len(validation["threshold"]))
    ))
    write_csv(out / "error_groups.csv", tuple(groups[0]), groups)
    write_csv(out / "misclassified_test.csv",
              ("error", "source_record", "actual", "predicted", "log_odds",
               "top_feature", "top_value", "top_log_odds_contribution", *rows[0]), errors)
    plot_results(out, curves, validation, threshold, matrices, groups, summary["splits"]["test"]["yes_rate"])
    write_report(out / "RESULTS.md", summary, groups, errors, path.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="bank+marketing.zip、bank.zip 或 bank-full.csv")
    parser.add_argument("--out", type=Path, default=Path("results_a"), help="结果目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-duration", action="store_true", help="仅作通话结束后的对照实验")
    parser.add_argument("--var-smoothing", type=float, default=0.01)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed 必须为非负整数。")
    try:
        summary = run(args.data, args.out, args.seed, args.include_duration, args.var_smoothing)
    except (OSError, ValueError, BadZipFile) as exc:
        parser.exit(1, f"错误：{exc}\n")
    print(json.dumps(summary["test"], indent=2))
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
