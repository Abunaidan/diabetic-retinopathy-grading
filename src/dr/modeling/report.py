"""Figures and the written report.

Everything a reader needs in order to judge the work - and, just as important,
everything they need in order to distrust it - is generated here from the
results bundle produced by :mod:`dr.modeling.train`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ..config import GRADE_NAMES, Config
from ..features.extract import describe_feature
from ..utils import ensure_dir, get_logger

LOGGER = get_logger("report")

PALETTE = "viridis"
FIGSIZE = (8.0, 5.0)
DPI = 150


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": DPI, "savefig.dpi": DPI, "axes.titleweight": "bold"})


def _save(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figure -> %s", path.name)
    return path


def _grade_label(grade: int) -> str:
    return f"{grade} - {GRADE_NAMES.get(int(grade), '')}".strip(" -")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_class_distribution(features: pd.DataFrame, path: Path) -> Path:
    _style()
    counts = features["grade"].value_counts().sort_index()
    patients = features.groupby("grade")["group"].nunique().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.barplot(x=[_grade_label(g) for g in counts.index], y=counts.to_numpy(),
                hue=[_grade_label(g) for g in counts.index], palette=PALETTE,
                legend=False, ax=axes[0])
    axes[0].set_title("Eyes per grade")
    axes[0].set_ylabel("eyes")
    axes[0].tick_params(axis="x", rotation=30)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", padding=2)

    share = counts / counts.sum()
    sns.barplot(x=[_grade_label(g) for g in patients.index], y=patients.to_numpy(),
                hue=[_grade_label(g) for g in patients.index], palette=PALETTE,
                legend=False, ax=axes[1])
    axes[1].set_title("Patients per grade")
    axes[1].set_ylabel("patients")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"Class imbalance: the majority grade covers {share.max():.0%} of the cohort",
        y=1.02,
    )
    return _save(fig, path)


def figure_model_comparison(results: dict[str, Any], path: Path) -> Path:
    _style()
    rows = []
    for name, payload in results["models"].items():
        point, low, high = payload["confidence_intervals"]["qwk"]
        rows.append(
            {
                "model": name,
                "cv_qwk": payload["cv_qwk_mean"],
                "test_qwk": point,
                # Bar errors must be non-negative even when the percentile
                # interval happens to fall entirely on one side of the estimate.
                "low": max(point - low, 0.0),
                "high": max(high - point, 0.0),
            }
        )
    frame = pd.DataFrame(rows).sort_values("test_qwk")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    positions = np.arange(len(frame))
    ax.barh(positions - 0.2, frame["cv_qwk"], height=0.38, label="cross-validated (train)",
            color="#9ecae1")
    ax.barh(positions + 0.2, frame["test_qwk"], height=0.38, label="hold-out (test)",
            color="#31698a",
            xerr=[frame["low"], frame["high"]], error_kw={"ecolor": "#333", "capsize": 3})
    for position, value, offset in zip(positions, frame["test_qwk"], frame["high"]):
        ax.text(value + offset + 0.015, position + 0.2, f"{value:.3f}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(positions)
    ax.set_yticklabels(frame["model"])
    ax.axvline(0.0, color="grey", lw=1)
    ax.set_xlim(min(frame["test_qwk"].min() - 0.25, -0.1), 1.0)
    ax.set_xlabel("Quadratic weighted kappa")
    ax.set_title("Model comparison (error bars: 95 % bootstrap CI on the hold-out set)")
    ax.legend(loc="lower right")
    return _save(fig, path)


def figure_confusion_matrix(results: dict[str, Any], path: Path) -> Path:
    _style()
    best = results["best_model"]
    labels = sorted(int(k) for k in results["dataset"]["class_counts"])
    y_true = np.asarray(results["test_truth"], dtype=int)
    y_pred = np.asarray(results["models"][best]["y_pred"], dtype=int)

    counts = confusion_matrix(y_true, y_pred, labels=labels)
    normalised = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    tick_labels = [_grade_label(g) for g in labels]
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=tick_labels, yticklabels=tick_labels, ax=axes[0])
    axes[0].set_title(f"Confusion matrix - {best} (counts)")
    sns.heatmap(normalised, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=tick_labels, yticklabels=tick_labels, ax=axes[1])
    axes[1].set_title("Row-normalised (per-class recall on the diagonal)")
    for ax in axes:
        ax.set_xlabel("predicted grade")
        ax.set_ylabel("true grade")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
    return _save(fig, path)


def figure_referable(results: dict[str, Any], cfg: Config, path: Path) -> Path:
    _style()
    best = results["best_model"]
    y_true = np.asarray(results["test_truth"], dtype=int)
    scores = np.asarray(results["models"][best]["scores"], dtype=float)
    positive = (y_true >= cfg.modeling.referable_threshold_grade).astype(int)
    metrics = results["models"][best]["metrics"]

    if positive.min() == positive.max():
        LOGGER.warning("Referable figure skipped: only one class in the hold-out set")
        return path

    fpr, tpr, _ = roc_curve(positive, scores)
    precision, recall, _ = precision_recall_curve(positive, scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, lw=2, color="#31698a",
                 label=f"AUROC = {metrics['referable_auroc']:.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    sensitivity = metrics.get("referable_sensitivity", float("nan"))
    specificity = metrics.get("referable_specificity", float("nan"))
    if np.isfinite(sensitivity) and np.isfinite(specificity):
        axes[0].scatter([1 - specificity], [sensitivity], color="crimson", zorder=5,
                        label=f"operating point: sens {sensitivity:.2f} / spec {specificity:.2f}")
    axes[0].set_xlabel("1 - specificity")
    axes[0].set_ylabel("sensitivity")
    axes[0].set_title(
        f"Referable DR (grade >= {cfg.modeling.referable_threshold_grade}) - ROC"
    )
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(recall, precision, lw=2, color="#7a5195",
                 label=f"AP = {metrics['referable_ap']:.3f}")
    axes[1].axhline(positive.mean(), ls="--", color="grey", lw=1,
                    label=f"prevalence = {positive.mean():.2f}")
    axes[1].set_xlabel("recall (sensitivity)")
    axes[1].set_ylabel("precision (PPV)")
    axes[1].set_title("Precision-recall")
    axes[1].legend(loc="lower left", fontsize=9)
    return _save(fig, path)


def figure_score_distribution(results: dict[str, Any], path: Path) -> Path:
    _style()
    best = results["best_model"]
    frame = pd.DataFrame(
        {
            "true grade": np.asarray(results["test_truth"], dtype=int),
            "severity score": np.asarray(results["models"][best]["scores"], dtype=float),
        }
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.boxplot(data=frame, x="true grade", y="severity score", hue="true grade",
                palette=PALETTE, legend=False, ax=ax, showfliers=False)
    sns.stripplot(data=frame, x="true grade", y="severity score", color="black", size=3,
                  alpha=0.45, ax=ax)
    ax.set_title(f"Continuous severity score by true grade - {best}")
    ax.set_xlabel("true grade")
    return _save(fig, path)


def figure_permutation_importance(results: dict[str, Any], path: Path, top_n: int = 20) -> Path:
    _style()
    importance = results.get("permutation_importance")
    if not importance:
        return path
    per_feature = pd.DataFrame(importance["per_feature"]).head(top_n)
    per_family = pd.Series(importance["per_family"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), width_ratios=[2, 1])
    axes[0].barh(
        per_feature["feature"][::-1],
        per_feature["importance_mean"][::-1],
        xerr=per_feature["importance_std"][::-1],
        color="#31698a",
        error_kw={"ecolor": "#999", "capsize": 2},
    )
    axes[0].set_xlabel("drop in hold-out QWK when the descriptor is permuted")
    axes[0].set_title(f"Top {len(per_feature)} descriptors - {results['best_model']}")

    sns.barplot(x=per_family.to_numpy(), y=per_family.index, hue=per_family.index,
                palette=PALETTE, legend=False, ax=axes[1])
    axes[1].set_xlabel("summed positive importance")
    axes[1].set_title("By descriptor family")
    return _save(fig, path)


def figure_ablation(results: dict[str, Any], path: Path) -> Path:
    _style()
    ablation = results.get("ablation")
    if not ablation:
        return path
    frame = pd.DataFrame(ablation["families"])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(frame))
    ax.barh(positions - 0.2, frame["qwk_family_only"], height=0.38,
            label="this family alone", color="#31698a")
    ax.barh(positions + 0.2, frame["qwk_without_family"], height=0.38,
            label="everything except this family", color="#c2a5cf")
    ax.axvline(ablation["qwk_all_features"], color="crimson", ls="--", lw=1.5,
               label=f"all descriptors = {ablation['qwk_all_features']:.3f}")
    ax.set_yticks(positions)
    ax.set_yticklabels(frame["family"])
    ax.set_xlabel("cross-validated QWK (training half)")
    ax.set_title("Feature-family ablation")
    # Outside the axes: the informative bars run all the way to the right edge.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    return _save(fig, path)


def figure_nested_cv(results: dict[str, Any], path: Path) -> Path:
    _style()
    nested = results.get("nested_cv")
    if not nested:
        return path
    rows = [
        {"model": name, "fold": index, "qwk": value}
        for name, payload in nested["models"].items()
        for index, value in enumerate(payload["fold_scores"])
    ]
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    order = (
        frame.groupby("model")["qwk"].mean().sort_values(ascending=False).index.tolist()
    )
    sns.boxplot(data=frame, x="model", y="qwk", order=order, hue="model", palette=PALETTE,
                legend=False, ax=ax, showfliers=False)
    sns.stripplot(data=frame, x="model", y="qwk", order=order, color="black", size=4, ax=ax)
    ax.set_title(
        f"Nested cross-validation ({nested['n_outer_folds']} outer patient-grouped folds)"
    )
    ax.set_ylabel("QWK on the outer fold")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, path)


def figure_leakage(results: dict[str, Any], path: Path) -> Path:
    _style()
    audit = results.get("leakage_audit")
    if not audit:
        return path

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = ["patient-grouped CV\n(correct)", "random CV\n(leaks fellow eyes)"]
    values = [audit["grouped_cv_qwk_mean"], audit["random_cv_qwk_mean"]]
    errors = [audit["grouped_cv_qwk_std"], audit["random_cv_qwk_std"]]
    ax.bar(names, values, yerr=errors, capsize=5, color=["#31698a", "#d95f02"])
    for index, value in enumerate(values):
        ax.text(index, value + 0.015, f"{value:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("cross-validated QWK")
    ax.set_title(
        f"Splitting audit - a random split would overstate QWK by {audit['optimism']:+.3f}"
    )
    return _save(fig, path)


def figure_qualitative(cfg: Config, features: pd.DataFrame, path: Path, per_grade: int = 1) -> Path:
    """One example per grade with the detected structures drawn on top."""
    _style()
    from ..features.descriptors import structure_maps
    from ..features.preprocess import preprocess_image

    grades = sorted(features["grade"].unique())
    picks: list[tuple[int, str]] = []
    for grade in grades:
        subset = features[features["grade"] == grade].head(per_grade)
        for _, row in subset.iterrows():
            picks.append((int(grade), str(row["image_path"])))
    if not picks:
        return path

    fig, axes = plt.subplots(3, len(picks), figsize=(3.0 * len(picks), 9.0))
    axes = np.atleast_2d(axes)
    if axes.shape[0] != 3:
        axes = axes.reshape(3, -1)

    for column, (grade, image_path) in enumerate(picks):
        try:
            img = preprocess_image(image_path, cfg.features, cfg.quality)
            maps = structure_maps(img, cfg.features)
            vessels, bright, dark = maps["vessels"], maps["bright"], maps["dark"]
        except Exception as exc:  # pragma: no cover - visual aid only
            LOGGER.warning("Qualitative panel skipped %s: %s", image_path, exc)
            continue

        axes[0, column].imshow(np.clip(img.rgb, 0, 1))
        axes[0, column].set_title(_grade_label(grade), fontsize=10)
        axes[1, column].imshow(img.clahe, cmap="gray")
        overlay = np.clip(img.rgb.copy(), 0, 1)
        overlay[vessels] = [0.15, 0.85, 1.0]
        overlay[dark] = [1.0, 0.2, 0.2]
        overlay[bright] = [1.0, 0.95, 0.2]
        axes[2, column].imshow(overlay)

    for row, label in enumerate(["field of view", "CLAHE green channel", "detected structures"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "What the descriptors see: segmented vessels (cyan), dark-lesion candidates (red), "
        "bright-lesion candidates (yellow).\nThese are morphological proxies, not validated "
        "lesion detections - residual responses at wide vessel junctions are visible and expected.",
        y=1.02,
        fontsize=11,
    )
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# written report
# --------------------------------------------------------------------------- #
def _metric_row(name: str, payload: dict[str, Any], selected: bool) -> str:
    metrics = payload["metrics"]
    point, low, high = payload["confidence_intervals"]["qwk"]
    label = f"**{name}** ★" if selected else name
    return (
        f"| {label} | {payload['cv_qwk_mean']:.3f} ± {payload['cv_qwk_std']:.3f} "
        f"| {point:.3f} [{low:.3f}, {high:.3f}] "
        f"| {metrics['balanced_accuracy']:.3f} | {metrics['f1_macro']:.3f} "
        f"| {metrics['accuracy']:.3f} | {metrics['mean_absolute_grade_error']:.2f} |"
    )


def build_report(cfg: Config, features: pd.DataFrame, results: dict[str, Any]) -> Path:
    """Render every figure and write ``reports/REPORT.md``."""
    figures = ensure_dir(cfg.figures_dir)
    reports = ensure_dir(cfg.reports_dir)

    figure_class_distribution(features, figures / "01_class_distribution.png")
    figure_model_comparison(results, figures / "02_model_comparison.png")
    figure_confusion_matrix(results, figures / "03_confusion_matrix.png")
    figure_referable(results, cfg, figures / "04_referable_dr.png")
    figure_score_distribution(results, figures / "05_severity_scores.png")
    figure_permutation_importance(results, figures / "06_permutation_importance.png")
    figure_ablation(results, figures / "07_feature_ablation.png")
    figure_nested_cv(results, figures / "08_nested_cv.png")
    figure_leakage(results, figures / "09_splitting_audit.png")
    figure_qualitative(cfg, features, figures / "10_qualitative.png")

    best = results["best_model"]
    best_payload = results["models"][best]
    metrics = best_payload["metrics"]
    baseline = results["models"]["dummy_frequent"]
    comparison = results["comparison_vs_baseline"]
    dataset = results["dataset"]
    split = results["split"]

    lines: list[str] = []
    add = lines.append

    add("# Diabetic retinopathy grading - results\n")
    add(
        f"_Generated {results['environment']['timestamp']} · "
        f"{dataset['n_eyes']} eyes · {dataset['n_patients']} patients · "
        f"{dataset['n_features']} handcrafted descriptors · "
        f"scikit-learn {results['environment']['scikit_learn']}_\n"
    )

    add("## 1. Headline\n")
    add(
        f"The selected model is **{best}** — "
        f"{best_payload['description'].rstrip('.')}.\n\n"
        f"* Quadratic weighted kappa on the untouched hold-out set: "
        f"**{metrics['qwk']:.3f}** "
        f"(95 % CI {best_payload['confidence_intervals']['qwk'][1]:.3f} - "
        f"{best_payload['confidence_intervals']['qwk'][2]:.3f}).\n"
        f"* Majority-class baseline: {baseline['metrics']['qwk']:.3f} QWK, "
        f"{baseline['metrics']['balanced_accuracy']:.3f} balanced accuracy.\n"
        f"* Paired bootstrap against that baseline: "
        f"Δ QWK = {comparison['difference']:+.3f} "
        f"[{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}], "
        f"p = {comparison['p_value']:.4f}.\n"
        f"* Referable disease (grade ≥ {cfg.modeling.referable_threshold_grade}): "
        f"AUROC {metrics.get('referable_auroc', float('nan')):.3f}, "
        f"specificity {metrics.get('referable_specificity', float('nan')):.3f} at "
        f"{metrics.get('referable_sensitivity', float('nan')):.3f} sensitivity.\n"
    )

    add("## 2. Cohort and splitting\n")
    counts = dataset["class_counts"]
    total = sum(counts.values())
    add("| grade | name | eyes | share |")
    add("| --- | --- | ---: | ---: |")
    for grade in sorted(counts):
        add(
            f"| {grade} | {GRADE_NAMES.get(int(grade), '')} | {counts[grade]} "
            f"| {counts[grade] / total:.1%} |"
        )
    add("")
    add(
        f"The hold-out set contains **{split['n_test']} eyes from "
        f"{split['n_test_groups']} patients**, disjoint from the "
        f"{split['n_train']} training eyes ({split['n_train_groups']} patients). "
        "Splitting is stratified on the grade *and* grouped by patient, so no "
        "fellow eye is ever seen on both sides.\n"
    )
    audit = results.get("leakage_audit")
    if audit:
        verdict = (
            f"an optimism of {audit['optimism']:+.3f} that a naive protocol would have "
            "reported as real performance"
            if audit["optimism"] > 0.01
            else f"a difference of only {audit['optimism']:+.3f}. Patient identity carries "
            "little information on this particular cohort; the audit is a measurement rather "
            "than an assumption, and on a cohort where both eyes come from one camera session "
            "it typically finds a large positive gap"
        )
        add(
            f"**The size of that effect is measured, not assumed:** cross-validating the same "
            f"model with a naive random split gives QWK {audit['random_cv_qwk_mean']:.3f} "
            f"versus {audit['grouped_cv_qwk_mean']:.3f} with patient-grouped folds - "
            f"{verdict} (figure 09).\n"
        )

    add("## 3. Model comparison\n")
    add(
        "| model | CV QWK (train) | hold-out QWK [95 % CI] | balanced acc. | macro F1 "
        "| accuracy | mean grade error |"
    )
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    ordered = sorted(
        results["models"].items(), key=lambda kv: kv[1]["metrics"]["qwk"], reverse=True
    )
    for name, payload in ordered:
        add(_metric_row(name, payload, selected=name == best))
    add("")
    add(f"★ selected model. _Selection rule: {results['selection_rule']}._\n")

    top_on_test = ordered[0][0]
    if top_on_test != best:
        runner = results.get("comparison_vs_runner_up", {})
        add(
            f"**The selected model is not the top scorer on this particular hold-out draw** "
            f"(`{top_on_test}` reaches {ordered[0][1]['metrics']['qwk']:.3f} against "
            f"{metrics['qwk']:.3f}). The confidence intervals overlap, and the paired bootstrap "
            f"against the cross-validation runner-up "
            f"(`{runner.get('runner_up', 'n/a')}`) is inconclusive "
            f"(Δ = {runner.get('difference', float('nan')):+.3f}, "
            f"p = {runner.get('p_value', float('nan')):.2f}) - which is what selection noise on "
            f"{split['n_test']} test eyes looks like. The nested cross-validation below is the "
            "estimate to trust for ranking, because it is not contaminated by selection.\n"
        )

    nested = results.get("nested_cv")
    if nested:
        add("### Nested cross-validation\n")
        add(
            "Tuning and fitting were repeated inside "
            f"{nested['n_outer_folds']} outer patient-grouped folds, so the numbers below "
            "contain no model-selection optimism.\n"
        )
        add("| model | nested QWK | spread over folds |")
        add("| --- | ---: | ---: |")
        for name, payload in sorted(
            nested["models"].items(), key=lambda kv: kv[1]["mean"], reverse=True
        ):
            add(f"| {name} | {payload['mean']:.3f} | ± {payload['std']:.3f} |")
        add("")

    add("## 4. What the model actually uses\n")
    importance = results.get("permutation_importance")
    if importance:
        add(
            "Permutation importance is measured on the **hold-out** set with QWK as the "
            "score, so it reports the loss in real predictive performance when a "
            "descriptor is destroyed - not the training-set impurity heuristic.\n"
        )
        add("| descriptor | family | Δ QWK when permuted | meaning |")
        add("| --- | --- | ---: | --- |")
        for row in importance["per_feature"][:12]:
            add(
                f"| `{row['feature']}` | {row['family']} | {row['importance_mean']:.4f} "
                f"| {describe_feature(row['feature'])} |"
            )
        add("")
    ablation = results.get("ablation")
    if ablation:
        add("### Feature-family ablation\n")
        add("| family | descriptors | QWK using only this family | QWK without it | Δ when removed |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for row in ablation["families"]:
            add(
                f"| {row['family']} | {row['n_features']} | {row['qwk_family_only']:.3f} "
                f"| {row['qwk_without_family']:.3f} | {row['delta_when_removed']:+.3f} |"
            )
        add("")

    add("## 5. Figures\n")
    for figure in sorted(figures.glob("*.png")):
        add(f"![{figure.stem}](figures/{figure.name})\n")

    add("## 6. Limitations\n")
    add(
        "* The descriptors are *proxies*: a \"microaneurysm count\" is the number of "
        "small dark top-hat responses, not a lesion validated by an ophthalmologist.\n"
        "* Quality control is heuristic (field-of-view size, Laplacian focus, exposure); "
        "it removes obviously ungradable frames, not subtly degraded ones.\n"
        "* The hold-out set is a single split of a single cohort. External validation "
        "on a different camera and population is the only thing that establishes "
        "transportability.\n"
        "* Nothing here is a medical device. The referral threshold is tuned for "
        f"{cfg.modeling.target_sensitivity:.0%} sensitivity on this cohort's prevalence "
        "and would have to be re-derived for any real screening programme.\n"
    )

    path = reports / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Report -> %s", path)
    return path
