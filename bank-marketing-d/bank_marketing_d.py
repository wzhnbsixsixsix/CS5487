"""CS5487 group project, member D — class-imbalance sensitivity study.

Reuses member A's split and preprocessing, member B's logistic regression and
member C's PCA/SVD so that every model is scored on the *same* test records.
No file outside ``bank-marketing-d/results_d/`` is written.

Stage 1 (``--stage scores``) builds the unified score table ``test_scores.csv``.
Every later experiment is derived from that table alone, so a single refit
reproduces the whole study.

Stage 2 (``--stage all``) runs the two sensitivity experiments:
    S1  operating-point / cost sensitivity
    S3  positive-prior (prevalence) shift sensitivity
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
A_DIR = PROJECT_DIR / "bank-marketing-a"
B_DIR = PROJECT_DIR / "bank-marketing-b"
C_DIR = PROJECT_DIR / "bank-marketing-c"
for _path in (A_DIR, B_DIR, C_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bank_marketing_a as a  # noqa: E402
import bank_marketing_b as b  # noqa: E402
import cs5487_c_group as c  # noqa: E402

SEED = 42
SPLIT_CSV = A_DIR / "results_a" / "split_indices.csv"
K_COMPONENTS = 45  # member C selected k = 45 on the validation split
DEFAULT_LAMBDA = 10.0  # member B selected lambda = 10 by mean fold PR-AUC

MODEL_NAMES = ("gnb", "lr_mle", "lr_map", "pca_gnb", "svd_gnb")
MODEL_LABELS = {
    "gnb": "Gaussian NB (A)",
    "lr_mle": "Logistic MLE (B)",
    "lr_map": "Logistic MAP (B)",
    "pca_gnb": "PCA-45 + NB (C)",
    "svd_gnb": "SVD-45 + NB (C)",
}

# Numbers published by A/B/C. Rounded to 6 dp; the score table must reproduce them.
EXPECTED = {
    "gnb": {"roc_auc": 0.749715, "pr_auc": 0.359247},
    "lr_mle": {"roc_auc": 0.763849, "pr_auc": 0.403700},
    "lr_map": {"roc_auc": 0.764123, "pr_auc": 0.403385},
    "pca_gnb": {"roc_auc": 0.733655, "pr_auc": 0.331495},
    "svd_gnb": {"roc_auc": 0.728579, "pr_auc": 0.321959},
}


# --------------------------------------------------------------------------- io


def read_split(path: Path, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read A's split file and verify it covers every record exactly once."""
    buckets: dict[str, list[int]] = {name: [] for name in a.SPLIT_NAMES}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["split"]
            if name not in buckets:
                raise ValueError(f"未知的划分名称：{name}")
            index = int(row["source_record"]) - 1
            if not 0 <= index < len(labels):
                raise ValueError(f"记录号越界：{row['source_record']}")
            if int(row["y"]) != int(labels[index]):
                raise ValueError(f"划分文件与数据的标签不一致：第 {row['source_record']} 条")
            buckets[name].append(index)
    parts = tuple(np.asarray(buckets[name], dtype=int) for name in a.SPLIT_NAMES)
    if sum(len(part) for part in parts) != len(labels):
        raise ValueError("划分文件未覆盖全部记录。")
    if len({i for part in parts for i in part}) != len(labels):
        raise ValueError("划分文件存在重复记录。")
    return parts


def write_csv(path: Path, columns, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


# ------------------------------------------------------------------------ fit


def fit_all_models(rows, labels, split, lam: float, k: int) -> dict[str, dict]:
    """Fit the five published models on A's training rows only.

    Returns ``{model: {"val": scores, "test": scores}}`` in log-odds space.
    """
    train, validation, test = split
    train_rows = [rows[i] for i in train]
    y_train = labels[train]

    preprocessor = a.BankPreprocessor().fit(train_rows)
    x_train, x_val, x_test = (
        preprocessor.transform([rows[i] for i in part]) for part in split
    )

    scores: dict[str, dict] = {}

    # A: handwritten Gaussian NB on the full 50-column encoding.
    gnb = a.GaussianNB().fit(x_train, y_train)
    scores["gnb"] = {"val": gnb.decision_function(x_val), "test": gnb.decision_function(x_test)}

    # B: logistic regression on 41 columns (first level of each field is the reference).
    design = b.LogisticDesign(train_rows)
    lr_train, lr_val, lr_test = (
        design.transform([rows[i] for i in part]) for part in split
    )
    for name, precision in (("lr_mle", 0.0), ("lr_map", lam)):
        fit = b.fit_logistic(lr_train, y_train, precision)
        scores[name] = {
            "val": b.decision_function(lr_val, fit),
            "test": b.decision_function(lr_test, fit),
        }

    # C: the same NB on 45 PCA / SVD components fitted on the training matrix only.
    for name, reducer in (
        ("pca_gnb", c.ManualPCA(n_components=k).fit(x_train)),
        ("svd_gnb", c.ManualSVDReducer(k).fit(x_train)),
    ):
        model = a.GaussianNB().fit(reducer.transform(x_train), y_train)
        scores[name] = {
            "val": model.decision_function(reducer.transform(x_val)),
            "test": model.decision_function(reducer.transform(x_test)),
        }

    return scores


def model_metrics(scores: dict[str, dict], labels, split) -> dict[str, dict]:
    """Test-set ranking metrics plus the validation-selected F1 threshold."""
    _, validation, test = split
    summary = {}
    for name in MODEL_NAMES:
        val_curve = a.ranking_curves(labels[validation], scores[name]["val"])
        threshold = a.select_threshold(val_curve)
        test_curve = a.ranking_curves(labels[test], scores[name]["test"])
        matrix = a.confusion_matrix(labels[test], (scores[name]["test"] >= threshold).astype(np.int8))
        summary[name] = {
            "roc_auc": test_curve["roc_auc"],
            "pr_auc": test_curve["pr_auc"],
            "average_precision": test_curve["average_precision"],
            "validation_best_f1": float(val_curve["f1"].max()),
            "validation_selected_threshold": threshold,
            "probability_threshold": float(a.sigmoid(np.asarray([threshold]))[0]),
            "confusion_matrix": matrix.tolist(),
            **a.classification_metrics(matrix),
        }
    return summary


def check_against_published(metrics: dict[str, dict]) -> list[dict]:
    """Compare the refit against A/B/C's published numbers at 6 decimal places."""
    checks = []
    for name, expected in EXPECTED.items():
        for key, want in expected.items():
            got = metrics[name][key]
            checks.append({
                "model": name,
                "metric": key,
                "published": want,
                "reproduced": round(got, 6),
                "delta": got - want,
                "match": round(got, 6) == want,
            })
    return checks


# --------------------------------------------------------------------- stages


def stage_scores(data_path: Path, out: Path, lam: float, k: int) -> dict:
    rows, labels = a.load_bank_data(data_path)
    split = read_split(SPLIT_CSV, labels)
    scores = fit_all_models(rows, labels, split, lam, k)
    metrics = model_metrics(scores, labels, split)
    checks = check_against_published(metrics)

    train, validation, test = split
    for part_name, key, part in (("validation", "val", validation), ("test", "test", test)):
        write_csv(
            out / f"{part_name}_scores.csv",
            ("source_record", "y_true", *MODEL_NAMES),
            (
                {"source_record": int(i) + 1, "y_true": int(labels[i]),
                 **{model: float(scores[model][key][n]) for model in MODEL_NAMES}}
                for n, i in enumerate(part)
            ),
        )

    summary = {
        "data": {"file": Path(data_path).name, "n_records": len(rows)},
        "split": {name: {"n": len(part), "positives": int(labels[part].sum())}
                  for name, part in zip(a.SPLIT_NAMES, split)},
        "settings": {"lambda_map": lam, "k_components": k, "seed": SEED,
                     "duration_included": False,
                     "split_source": str(SPLIT_CSV.relative_to(PROJECT_DIR))},
        "test_metrics": metrics,
        "reproduction_check": checks,
        "reproduction_ok": all(check["match"] for check in checks),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ------------------------------------------------- S1: operating-point sensitivity

COLORS = {
    "gnb": "#5b6b7d",
    "lr_mle": "#29649c",
    "lr_map": "#137d69",
    "pca_gnb": "#b4612a",
    "svd_gnb": "#8b4a8f",
}
COST_RATIOS = np.round(np.arange(1.0, 20.0001, 0.25), 4)
REPORT_RATIOS = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0)


def read_score_table(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    labels = np.asarray([int(r["y_true"]) for r in records])
    scores = {name: np.asarray([float(r[name]) for r in records]) for name in MODEL_NAMES}
    return labels, scores


def _segments(winners: list[str]) -> list[dict]:
    """Collapse a per-ratio winner sequence into contiguous ranges."""
    segments, start = [], 0
    for i in range(1, len(winners)):
        if winners[i] != winners[start]:
            segments.append({"winner": winners[start],
                             "ratio_from": float(COST_RATIOS[start]),
                             "ratio_to": float(COST_RATIOS[i - 1])})
            start = i
    segments.append({"winner": winners[start],
                     "ratio_from": float(COST_RATIOS[start]),
                     "ratio_to": float(COST_RATIOS[-1])})
    return segments


def _plot_f1(out: Path, curves, thresholds, labels, scores) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    for name in MODEL_NAMES:
        curve = curves[name]
        probability = a.sigmoid(curve["threshold"])
        order = np.argsort(probability)
        ax.plot(probability[order], curve["f1"][order], color=COLORS[name],
                linewidth=1.7, label=MODEL_LABELS[name])
        threshold = thresholds[name]
        predicted = (scores[name] >= threshold).astype(np.int8)
        f1_at = a.classification_metrics(a.confusion_matrix(labels, predicted))["f1"]
        marker = float(a.sigmoid(np.asarray([threshold]))[0])
        ax.plot([marker], [f1_at], marker="o", markersize=7, color=COLORS[name],
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 0.62)
    ax.set_xlabel("decision threshold (probability)")
    ax.set_ylabel("F1 on the test set")
    ax.set_title("Where the cut line was placed")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.annotate("NB family: F1 barely moves\nbetween 0.2 and 0.8",
                xy=(0.60, 0.383), xytext=(0.52, 0.30),
                fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.8))
    ax.annotate("LR: a narrow peak,\nthen it collapses",
                xy=(0.30, 0.38), xytext=(0.30, 0.47),
                fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.8))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"s1_f1_vs_threshold.{suffix}", dpi=300)
    plt.close(fig)


def _plot_cost(out: Path, best_cost, n: int, n_pos: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), layout="constrained")
    for name in MODEL_NAMES:
        ax.plot(COST_RATIOS, best_cost[name], color=COLORS[name],
                linewidth=1.8, label=MODEL_LABELS[name])
    ax.plot(COST_RATIOS, COST_RATIOS * n_pos / n * 1000.0, color="#999999",
            linestyle=":", linewidth=1.2, label="call nobody")
    ax.plot(COST_RATIOS, np.full_like(COST_RATIOS, (n - n_pos) / n * 1000.0),
            color="#c0392b", linestyle="--", linewidth=1.2, label="call everybody")

    ax.set_xlabel("cost of one missed subscriber / cost of one wasted call")
    ax.set_ylabel("minimum cost per 1,000 clients")
    ax.set_title("Cheapest achievable cost as the missed-subscriber price grows")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    ax.annotate("all five models overlap\nwithin 0.65% of each other",
                xy=(13.0, 690.0), xytext=(9.2, 380.0),
                fontsize=7.5, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.8))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"s1_cost_vs_ratio.{suffix}", dpi=300)
    plt.close(fig)


def stage_s1(out: Path) -> dict:
    """Operating-point and cost sensitivity, computed from the score table alone."""
    labels, scores = read_score_table(out / "test_scores.csv")
    val_labels, val_scores = read_score_table(out / "validation_scores.csv")
    n, n_pos = len(labels), int(labels.sum())
    n_neg = n - n_pos

    curves = {name: a.ranking_curves(labels, scores[name]) for name in MODEL_NAMES}
    thresholds = {name: a.select_threshold(a.ranking_curves(val_labels, val_scores[name]))
                  for name in MODEL_NAMES}

    operating = []
    fixed_cost = {}
    for name in MODEL_NAMES:
        threshold = thresholds[name]
        predicted = (scores[name] >= threshold).astype(np.int8)
        matrix = a.confusion_matrix(labels, predicted)
        tn, fp, fn, tp = map(int, matrix.ravel())
        fixed_cost[name] = (fp + COST_RATIOS * fn) / n * 1000.0
        operating.append({
            "model": name,
            "threshold_log_odds": threshold,
            "threshold_probability": float(a.sigmoid(np.asarray([threshold]))[0]),
            "predicted_positive": tp + fp,
            "positive_share": (tp + fp) / n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            **a.classification_metrics(matrix),
        })
    write_csv(out / "s1_operating_points.csv", tuple(operating[0]), operating)

    best_cost, best_threshold = {}, {}
    for name in MODEL_NAMES:
        curve = curves[name]
        grid = (curve["fp"][None, :] + COST_RATIOS[:, None] * curve["fn"][None, :]) / n * 1000.0
        index = grid.argmin(axis=1)
        best_cost[name] = grid.min(axis=1)
        best_threshold[name] = curve["threshold"][index]

    table = []
    for i, ratio in enumerate(COST_RATIOS):
        table.append({
            "cost_ratio": float(ratio),
            **{f"cost_best_{name}": float(best_cost[name][i]) for name in MODEL_NAMES},
            **{f"cost_fixed_{name}": float(fixed_cost[name][i]) for name in MODEL_NAMES},
            **{f"threshold_best_{name}": float(best_threshold[name][i]) for name in MODEL_NAMES},
        })
    write_csv(out / "s1_cost_curves.csv", tuple(table[0]), table)

    best_winners = [min(MODEL_NAMES, key=lambda m: (best_cost[m][i], m)) for i in range(len(COST_RATIOS))]
    fixed_winners = [min(MODEL_NAMES, key=lambda m: (fixed_cost[m][i], m)) for i in range(len(COST_RATIOS))]

    _plot_f1(out, curves, thresholds, labels, scores)
    _plot_cost(out, best_cost, n, n_pos)

    team_pick = max(operating, key=lambda row: row["f1"])["model"]

    headline = []
    for ratio in REPORT_RATIOS:
        i = int(np.flatnonzero(np.isclose(COST_RATIOS, ratio))[0])
        best_any = float(min(best_cost[name][i] for name in MODEL_NAMES))
        team_cost = float(fixed_cost[team_pick][i])
        nobody = float(ratio * n_pos / n * 1000.0)
        headline.append({
            "cost_ratio": ratio,
            "best_model": best_winners[i],
            "best_cost_per_1000": best_any,
            "fixed_threshold_model": fixed_winners[i],
            "fixed_threshold_cost_per_1000": float(fixed_cost[fixed_winners[i]][i]),
            "team_pick": team_pick,
            "team_pick_cost_per_1000": team_cost,
            "team_pick_overspend_vs_best_pct": (team_cost / best_any - 1.0) * 100.0,
            "call_everybody_cost_per_1000": float((n - n_pos) / n * 1000.0),
            "call_nobody_cost_per_1000": nobody,
            "best_vs_call_nobody_pct": (1.0 - best_any / nobody) * 100.0,
        })
    write_csv(out / "s1_flip_table.csv", tuple(headline[0]), headline)

    summary = {
        "n_test": n, "n_positive": n_pos, "prevalence": n_pos / n,
        "operating_points": operating,
        "best_threshold_segments": _segments(best_winners),
        "fixed_threshold_segments": _segments(fixed_winners),
        "headline": headline,
        "notes": ("cost = FP + ratio*FN per 1,000 clients. 'best' re-picks the threshold for "
                  "each ratio; 'fixed' keeps the validation-selected F1 threshold."),
    }
    (out / "s1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"flip_summary": {"best_threshold_segments": summary["best_threshold_segments"],
                             "headline": headline}}


# -------------------------------------------------------- S3: prior-shift sensitivity

PREVALENCE_GRID = np.round(np.arange(0.02, 0.6001, 0.01), 4)


def _weighted_curves(labels_sorted, weights_sorted, ends, scores_sorted):
    """Weighted ROC / PR areas and the weighted F1 at every threshold."""
    tp = np.r_[0.0, np.cumsum(weights_sorted * labels_sorted)[ends]]
    fp = np.r_[0.0, np.cumsum(weights_sorted * (1.0 - labels_sorted))[ends]]
    positives, negatives = tp[-1], fp[-1]
    fn = positives - tp
    recall = tp / positives
    fpr = fp / negatives
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    precision[0] = 1.0
    roc = float((np.diff(fpr) * (recall[1:] + recall[:-1]) * 0.5).sum())
    pr = float((np.diff(recall) * (precision[1:] + precision[:-1]) * 0.5).sum())
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp),
                   where=(2 * tp + fp + fn) > 0)
    return roc, pr, f1


def _plot_s3(out: Path, frames, base: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.6), layout="constrained")
    styles = {"lr_mle": "--"}
    for name in MODEL_NAMES:
        frame = frames[name]
        style = styles.get(name, "-")
        left.plot(frame["prevalence"], frame["roc_auc"], color=COLORS[name],
                  linewidth=1.8, linestyle=style, label=MODEL_LABELS[name])
        right.plot(frame["prevalence"], frame["pr_auc"], color=COLORS[name],
                   linewidth=1.8, linestyle=style, label=MODEL_LABELS[name])
    right.plot(frame["prevalence"], frame["baseline"], color="#999999", linestyle="--",
               linewidth=1.3, label="random baseline = positive rate")

    for axis in (left, right):
        axis.axvline(base, color="#c0392b", linestyle=":", linewidth=1.1)
        axis.set_xlabel("assumed positive rate")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_xlim(float(PREVALENCE_GRID[0]), float(PREVALENCE_GRID[-1]))

    left.set_ylabel("ROC-AUC")
    left.set_ylim(0.70, 0.79)
    left.set_title("ROC-AUC: unchanged by construction", fontsize=10)
    left.legend(fontsize=7, loc="lower right", frameon=False)

    right.set_ylabel("PR-AUC")
    right.set_ylim(0.0, 0.95)
    right.set_title("PR-AUC: moves with the positive rate", fontsize=10)
    right.legend(fontsize=7, loc="lower right", frameon=False)
    right.annotate(f"this split {base:.1%}", xy=(base, 0.86), xytext=(base + 0.035, 0.86),
                   fontsize=7, color="#c0392b", va="center",
                   arrowprops=dict(arrowstyle="->", color="#c0392b", linewidth=0.8))

    for suffix in ("png", "pdf"):
        fig.savefig(out / f"s3_prior_shift.{suffix}", dpi=300)
    plt.close(fig)


def stage_s3(out: Path) -> dict:
    """Prior-shift sensitivity: reweight the test set, never resample it."""
    labels, scores = read_score_table(out / "test_scores.csv")
    n = len(labels)
    base = float(labels.mean())

    order = {name: np.argsort(-scores[name], kind="stable") for name in MODEL_NAMES}
    ends = {name: np.r_[np.flatnonzero(np.diff(scores[name][order[name]]) != 0), n - 1]
            for name in MODEL_NAMES}
    threshold_grid = {name: np.r_[np.inf, scores[name][order[name]][ends[name]]]
                      for name in MODEL_NAMES}

    grid = np.unique(np.r_[PREVALENCE_GRID, base])
    frames, rows, threshold_rows = {}, [], []
    for name in MODEL_NAMES:
        y_sorted = labels[order[name]]
        probability = a.sigmoid(scores[name])
        records = []
        for target in grid:
            weight_pos, weight_neg = target / base, (1.0 - target) / (1.0 - base)
            weights = np.where(labels == 1, weight_pos, weight_neg)
            w_sorted = weights[order[name]]
            roc, pr, f1 = _weighted_curves(y_sorted, w_sorted, ends[name], scores[name][order[name]])
            brier = float((w_sorted * (probability[order[name]] - y_sorted) ** 2).sum() / w_sorted.sum())
            records.append({"prevalence": float(target), "roc_auc": roc, "pr_auc": pr,
                            "baseline": float(target),
                            "lift": pr / float(target) if target > 0 else float("nan"),
                            "brier": brier})
            best = 1 + int(np.argmax(f1[1:]))
            threshold_rows.append({
                "prevalence": float(target), "model": name,
                "f1_optimal_threshold_log_odds": float(threshold_grid[name][best]),
                "f1_optimal_threshold_probability": float(a.sigmoid(np.asarray([threshold_grid[name][best]]))[0]),
                "weighted_best_f1": float(f1[best]),
            })
        frames[name] = {key: np.asarray([r[key] for r in records]) for key in
                        ("prevalence", "roc_auc", "pr_auc", "baseline", "lift", "brier")}
        rows.extend(dict(model=name, **record) for record in records)

    write_csv(out / "s3_metrics.csv", ("model", "prevalence", "roc_auc", "pr_auc",
                                       "baseline", "lift", "brier"), rows)
    write_csv(out / "s3_thresholds.csv", tuple(threshold_rows[0]), threshold_rows)
    _plot_s3(out, frames, base)

    indices = {float(v): i for i, v in enumerate(grid)}
    threshold_lookup = {(row["prevalence"], row["model"]): row["f1_optimal_threshold_probability"]
                        for row in threshold_rows}
    reported = []
    for target in (0.05, base, 0.20, 0.30, 0.50):
        i = indices[target]
        for name in MODEL_NAMES:
            reported.append({
                "prevalence": target,
                "model": name,
                "roc_auc": float(frames[name]["roc_auc"][i]),
                "pr_auc": float(frames[name]["pr_auc"][i]),
                "lift": float(frames[name]["lift"][i]),
                "brier": float(frames[name]["brier"][i]),
                "f1_optimal_threshold_probability": threshold_lookup[(target, name)],
            })

    deviations = {name: float(np.ptp(frames[name]["roc_auc"])) for name in MODEL_NAMES}
    summary = {
        "base_prevalence": base,
        "roc_auc_spread_across_prevalences": deviations,
        "pr_auc_at_base": {name: float(frames[name]["pr_auc"][indices[base]])
                           for name in MODEL_NAMES},
        "reported": reported,
        "notes": ("Weights are p'/p for positives and (1-p')/(1-p) for negatives; no record is "
                  "added, removed or relabelled. ROC-AUC is invariant because both axes are "
                  "within-class rates; PR-AUC is not, because precision mixes both classes."),
    }
    (out / "s3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="bank+marketing.zip / bank.zip / bank-full.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results_d")
    parser.add_argument("--stage", choices=("scores", "s1", "s3", "all"), default="scores")
    parser.add_argument("--lambda-map", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--k", type=int, default=K_COMPONENTS)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.stage in ("scores", "all"):
        summary = stage_scores(args.data.resolve(), out, args.lambda_map, args.k)
        print(json.dumps(summary["reproduction_check"], indent=2))
        print("reproduction_ok:", summary["reproduction_ok"])

    if args.stage in ("s1", "all"):
        table = stage_s1(out)
        print(json.dumps(table["flip_summary"], indent=2))

    if args.stage in ("s3", "all"):
        summary = stage_s3(out)
        print(json.dumps({"base_prevalence": summary["base_prevalence"],
                          "roc_auc_spread": summary["roc_auc_spread_across_prevalences"]},
                         indent=2))

    print(f"Results: {out}")


if __name__ == "__main__":
    main()
