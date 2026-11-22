"""在成员 A 的划分上比较 Gaussian NB、逻辑回归 MLE 和高斯先验 MAP。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import scipy
from scipy.optimize import minimize

PROJECT_DIR = Path(__file__).resolve().parent
A_DIR = PROJECT_DIR.parent / "bank-marketing-a"
sys.path.insert(0, str(A_DIR))
from bank_marketing_a import (  # noqa: E402
    RATIOS, SPLIT_NAMES, BankPreprocessor, GaussianNB, classification_metrics,
    collect_errors, confusion_matrix, group_errors, load_bank_data, ranking_curves,
    select_threshold, sigmoid, stratified_split, write_csv,
)

SEED = 42
N_FOLDS = 5
LAMBDA_GRID = (0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
RANKING_KEYS = ("roc_auc", "pr_auc", "average_precision")


class LogisticDesign:
    """复用 A 的预处理，按字段剔除词表首个类别，作为截距模型的参考水平。"""

    def __init__(self, rows):
        self.encoder = BankPreprocessor().fit(rows)
        self.references = {field: levels[0] for field, levels in self.encoder.levels.items()}
        excluded = {f"{field}={level}" for field, level in self.references.items()}
        self.keep = [i for i, name in enumerate(self.encoder.feature_names) if name not in excluded]
        self.feature_names = [self.encoder.feature_names[i] for i in self.keep]

    def transform(self, rows):
        return self.encoder.transform(rows)[:, self.keep]


def objective(theta, x, y, precision):
    """总负对数似然与梯度，截距位于 theta[0]，先验仅作用于其余系数。"""
    beta = theta[1:]
    logits = theta[0] + x @ beta
    nll = np.logaddexp(0.0, (1 - 2 * y) * logits).sum()
    residual = sigmoid(logits) - y
    penalty = 0.5 * precision * (beta @ beta)
    gradient = np.r_[residual.sum(), x.T @ residual + precision * beta]
    return nll + penalty, gradient


def fit_logistic(x, y, precision):
    """使用手写目标及梯度，以 L-BFGS 求 MLE 或 MAP。"""
    result = minimize(
        objective, np.zeros(x.shape[1] + 1), args=(x, y, precision), jac=True,
        method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"逻辑回归求解失败（λ={precision:g}）：{result.message}")
    return result


def decision_function(x, fit):
    return fit.x[0] + x @ fit.x[1:]


def stratified_folds(y, seed):
    """折内索引相对于 A 的训练集，两个类别分别打乱和分配。"""
    rng = np.random.default_rng(seed)
    chunks = []
    for label in (0, 1):
        positions = rng.permutation(np.flatnonzero(y == label))
        if len(positions) < N_FOLDS:
            raise ValueError(f"每类至少需要 {N_FOLDS} 条训练记录。")
        chunks.append(np.array_split(positions, N_FOLDS))
    return [np.sort(np.r_[no, yes]) for no, yes in zip(*chunks)]


def cross_validate(rows, labels, folds):
    fold_results = {precision: [] for precision in LAMBDA_GRID}
    all_positions = np.arange(len(labels))
    for fold_ids in folds:
        fit_ids = np.setdiff1d(all_positions, fold_ids, assume_unique=True)
        fit_rows = [rows[i] for i in fit_ids]
        fold_rows = [rows[i] for i in fold_ids]
        design = LogisticDesign(fit_rows)
        x_fit, x_fold = design.transform(fit_rows), design.transform(fold_rows)
        for precision, results in fold_results.items():
            fit = fit_logistic(x_fit, labels[fit_ids], precision)
            curves = ranking_curves(labels[fold_ids], decision_function(x_fold, fit))
            results.append(curves["pr_auc"])
    table = [
        dict(lambda_precision=precision, mean_pr_auc=float(np.mean(scores)),
             sd_pr_auc=float(np.std(scores, ddof=1)),
             **{f"fold_{i}_pr_auc": score for i, score in enumerate(scores, 1)})
        for precision, scores in fold_results.items()
    ]
    # λ=0 是 MLE 对照。仅在正值中选 MAP，并列优先较弱先验。
    best = max(table[1:], key=lambda row: (row["mean_pr_auc"], -row["lambda_precision"]))
    return best["lambda_precision"], table


def check_a_split(splits, y):
    expected = ((name, int(i) + 1, int(y[i])) for name, ids in zip(SPLIT_NAMES, splits) for i in ids)
    with (A_DIR / "results_a" / "split_indices.csv").open(encoding="utf-8-sig", newline="") as stream:
        actual = [(row["split"], int(row["source_record"]), int(row["y"])) for row in csv.DictReader(stream)]
    if list(expected) != actual:
        raise ValueError("数据的记录顺序、标签或划分与 A 的 split_indices.csv 不一致。")


def data_fingerprint(rows, labels):
    """按原始行序对读入的字段和标签取哈希，兼容 ZIP、CSV 与换行差异。"""
    digest = hashlib.sha256()
    for row, label in zip(rows, labels):
        record = json.dumps([row, int(label)], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update((record + "\n").encode("utf-8"))
    return digest.hexdigest()


def evaluate(y_val, y_test, val_scores, test_scores):
    validation = ranking_curves(y_val, val_scores)
    threshold = select_threshold(validation)
    test = ranking_curves(y_test, test_scores)
    summary = dict(validation_best_f1=float(validation["f1"].max()),
                   ranking={key: test[key] for key in RANKING_KEYS})
    for name, cutoff in (("default", 0.0), ("validation_selected", threshold)):
        matrix = confusion_matrix(y_test, (test_scores >= cutoff).astype(np.int8))
        summary[name] = dict(log_odds_threshold=cutoff,
                             probability_threshold=float(sigmoid(cutoff)),
                             confusion_matrix=matrix.tolist(), **classification_metrics(matrix))
    return summary, test, (test_scores >= threshold).astype(np.int8)


def plot_comparison(out, curves, results, prevalence):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Gaussian NB": "#687788", "MLE": "#29649c", "MAP": "#137d69"}
    fig, (pr, counts) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    for name, curve in curves.items():
        linestyle = "--" if name == "MLE" else "-"
        pr.plot(curve["recall"], curve["precision"], color=colors[name], linestyle=linestyle,
                linewidth=1.8, label=f'{name} ({curve["pr_auc"]:.4f})')
    pr.axhline(prevalence, color="#89919a", linestyle=":", label=f"Prevalence ({prevalence:.1%})")
    pr.set(title="Precision–recall on the test set", xlabel="Recall", ylabel="Precision",
           xlim=(0, 1), ylim=(0, 1.04))
    pr.legend(frameon=False, fontsize=9)
    pr.grid(alpha=0.2)

    width = 0.23
    for i, (name, result) in enumerate(results.items()):
        (_, fp), (fn, _) = result["validation_selected"]["confusion_matrix"]
        bars = counts.bar(np.arange(2) + (i - 1) * width, [fp, fn], width, color=colors[name])
        counts.bar_label(bars, padding=3, fontsize=9)
    counts.set(title="Test errors at validation-selected thresholds", ylabel="Records")
    counts.set_xticks([0, 1], ["False positives", "False negatives"])
    counts.margins(y=0.14)
    counts.grid(axis="y", alpha=0.2)
    counts.set_axisbelow(True)
    fig.savefig(out / "test_comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    return matplotlib.__version__


def run(data, out):
    rows, labels = load_bank_data(data)
    train, val, test = splits = stratified_split(labels, seed=SEED)
    check_a_split(splits, labels)
    train_rows, val_rows, test_rows = ([rows[i] for i in ids] for ids in splits)
    folds = stratified_folds(labels[train], SEED)
    precision, cv = cross_validate(train_rows, labels[train], folds)

    design = LogisticDesign(train_rows)
    a_train, a_val, a_test = [design.encoder.transform(part) for part in (train_rows, val_rows, test_rows)]
    x_train, x_val, x_test = [matrix[:, design.keep] for matrix in (a_train, a_val, a_test)]
    baseline = GaussianNB().fit(a_train, labels[train])
    fits = {"MLE": fit_logistic(x_train, labels[train], 0.0),
            "MAP": fit_logistic(x_train, labels[train], precision)}
    scores = {"Gaussian NB": (baseline.decision_function(a_val), baseline.decision_function(a_test))}
    scores.update({name: (decision_function(x_val, fit), decision_function(x_test, fit))
                   for name, fit in fits.items()})

    # 保留原始行号和预测字段。duration 未进入任一模型，也无需进入错例表。
    case_rows = [{key: value for key, value in row.items() if key != "duration"} for row in test_rows]
    results, curves, predictions, groups, errors = {}, {}, {}, [], []
    for name, (val_scores, test_scores) in scores.items():
        results[name], curves[name], predictions[name] = evaluate(labels[val], labels[test], val_scores, test_scores)
        pred = predictions[name]
        if name == "Gaussian NB":
            columns, contributions = design.encoder.feature_names, baseline.feature_log_odds(a_test)
        else:
            columns, contributions = design.feature_names, x_test * fits[name].x[1:]
        groups.extend(dict(model=name, **row) for row in group_errors(case_rows, labels[test], pred))
        errors.extend(dict(model=name, **row) for row in collect_errors(
            case_rows, test, labels[test], pred, test_scores, columns, contributions,
        ))

    out.mkdir(parents=True, exist_ok=True)
    prevalence = float(labels[test].mean())
    matplotlib_version = plot_comparison(out, curves, results, prevalence)
    summary = dict(
        data=dict(source="https://doi.org/10.24432/C5K306", release="UCI Bank Marketing / bank-full.csv",
                  n=len(rows), canonical_records_sha256=data_fingerprint(rows, labels)),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, matplotlib=matplotlib_version),
        seed=SEED, split_ratios=RATIOS, include_duration=False, gaussian_nb_var_smoothing=baseline.var_smoothing,
        splits={name: dict(n=len(ids), yes=int(labels[ids].sum())) for name, ids in zip(SPLIT_NAMES, splits)},
        cv=dict(folds=N_FOLDS, metric="mean fold trapezoidal PR-AUC", lambda_grid=LAMBDA_GRID,
                selected_lambda=precision),
        n_logistic_features=len(design.feature_names), reference_levels=design.references,
        numeric_scaling={name: dict(mean=float(mean), scale=float(scale)) for name, mean, scale in
                         zip(design.encoder.numeric_columns, design.encoder.mean, design.encoder.scale)},
        fit={name: dict(iterations=int(fit.nit), objective=float(fit.fun),
                        gradient_inf_norm=float(np.linalg.norm(fit.jac, ord=np.inf)),
                        coefficient_l2_norm=float(np.linalg.norm(fit.x[1:])))
             for name, fit in fits.items()},
        test_mle_map_disagreements={str(label): {
            "mle_yes_map_no": int(np.count_nonzero((labels[test] == label) & (predictions["MLE"] > predictions["MAP"]))),
            "mle_no_map_yes": int(np.count_nonzero((labels[test] == label) & (predictions["MLE"] < predictions["MAP"]))),
        } for label in (0, 1)},
        test=results,
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(out / "cv_pr_auc.csv", tuple(cv[0]), cv)
    write_csv(out / "cv_fold_indices.csv", ("fold", "source_record", "y"), (
        dict(fold=f, source_record=int(train[i]) + 1, y=int(labels[train[i]]))
        for f, ids in enumerate(folds, 1) for i in ids
    ))
    write_csv(out / "coefficients.csv", ("feature", "mle", "map"), (
        dict(feature=name, mle=float(mle), map=float(map_))
        for name, mle, map_ in zip(("intercept", *design.feature_names), fits["MLE"].x, fits["MAP"].x)
    ))
    write_csv(out / "error_groups.csv", tuple(groups[0]), groups)
    write_csv(out / "misclassified_test.csv",
              ("model", "error", "source_record", "actual", "predicted", "log_odds",
               "top_feature", "top_value", "top_log_odds_contribution", *case_rows[0]), errors)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="bank+marketing.zip、bank.zip 或 bank-full.csv")
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "results_b", help="结果目录")
    args = parser.parse_args()
    try:
        summary = run(args.data, args.out)
    except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
        parser.exit(1, f"错误：{exc}\n")
    print(json.dumps(dict(cv=summary["cv"], test=summary["test"]), indent=2))
    print(f"Results: {args.out.resolve()}")


if __name__ == "__main__":
    main()
