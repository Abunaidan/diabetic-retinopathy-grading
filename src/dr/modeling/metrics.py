"""Metrics for an ordinal, heavily imbalanced, clinically consequential task.

Why not accuracy?  In a screening cohort roughly half the eyes are healthy, so a
model that always predicts "No DR" scores ~50 % accuracy while missing every
patient who needs treatment.  Three families of metric are used instead:

* **Quadratic weighted kappa (QWK)** - the primary metric.  It is the standard
  for retinopathy grading because it penalises a 0-vs-4 mistake sixteen times
  more than a 0-vs-1 mistake, which is exactly how the clinical cost behaves.
* **Balanced accuracy and macro F1** - class-symmetric sanity checks that keep
  the rare severe grades visible.
* **Referable-DR operating point** - the decision that is actually deployed:
  refer everyone with grade >= 2.  Reported as AUROC/average precision plus the
  specificity attainable at a clinically mandated sensitivity.

Every headline number is accompanied by a bootstrap confidence interval, and
model-versus-baseline comparisons use a paired bootstrap so that the difference
is judged on the same resampled patients.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "quadratic_weighted_kappa",
    "qwk_scorer",
    "balanced_accuracy",
    "macro_f1",
    "classification_metrics",
    "referable_metrics",
    "ordinal_scores",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "METRIC_LABELS",
]

def _confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Confusion matrix over a fixed label set, without scikit-learn's overhead.

    Bootstrapping calls the metrics tens of thousands of times; the validation
    and dtype machinery inside the scikit-learn entry points then dominates the
    runtime.  ``tests/test_metrics.py`` pins these fast paths to the reference
    implementations, so the speed costs nothing in trust.
    """
    n = len(labels)
    lookup = {int(label): index for index, label in enumerate(labels)}
    true_index = np.array([lookup.get(int(v), -1) for v in y_true])
    pred_index = np.array([lookup.get(int(v), -1) for v in y_pred])
    keep = (true_index >= 0) & (pred_index >= 0)
    matrix = np.bincount(
        true_index[keep] * n + pred_index[keep], minlength=n * n
    ).reshape(n, n)
    return matrix.astype(float)


def _balanced_accuracy_from_confusion(matrix: np.ndarray) -> float:
    """Mean per-class recall over the classes actually present in y_true."""
    support = matrix.sum(axis=1)
    present = support > 0
    if not present.any():
        return 0.0
    recall = np.diag(matrix)[present] / support[present]
    return float(recall.mean())


def _macro_f1_from_confusion(matrix: np.ndarray) -> float:
    """Macro F1 over the full label set, scoring absent classes as 0."""
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    denominator = predicted + actual
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denominator > 0, 2.0 * true_positive / denominator, 0.0)
    return float(f1.mean())


METRIC_LABELS: dict[str, str] = {
    "qwk": "Quadratic weighted kappa",
    "balanced_accuracy": "Balanced accuracy",
    "f1_macro": "Macro F1",
    "accuracy": "Accuracy",
    "referable_auroc": "Referable DR AUROC",
    "referable_ap": "Referable DR average precision",
    "referable_sensitivity": "Sensitivity at the operating point",
    "referable_specificity": "Specificity at the operating point",
}


# --------------------------------------------------------------------------- #
# quadratic weighted kappa
# --------------------------------------------------------------------------- #
def quadratic_weighted_kappa(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int] | None = None,
) -> float:
    """Cohen's kappa with quadratic weights.

    Implemented explicitly rather than called from scikit-learn so that the
    degenerate case (a rater that never varies, which happens for the
    most-frequent baseline) returns 0.0 instead of NaN and cannot poison a
    cross-validation average.  ``tests/test_metrics.py`` checks this
    implementation against ``sklearn.metrics.cohen_kappa_score`` on random data.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.size == 0:
        return 0.0

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(sorted(labels))
    n_labels = len(labels)
    if n_labels < 2:
        return 0.0

    observed = _confusion(y_true, y_pred, labels)
    observed /= max(observed.sum(), 1.0)

    index = np.arange(n_labels, dtype=float)
    weights = (index[:, None] - index[None, :]) ** 2 / (n_labels - 1) ** 2

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred)

    denominator = float((weights * expected).sum())
    if denominator <= 1e-12:
        return 0.0
    return float(1.0 - (weights * observed).sum() / denominator)


def make_qwk_scorer(labels: Sequence[int]) -> Callable:
    """Scorer bound to a fixed label set, safe for folds missing a rare grade."""
    return make_scorer(
        lambda y_true, y_pred: quadratic_weighted_kappa(y_true, y_pred, labels=labels),
        greater_is_better=True,
    )


qwk_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)


# --------------------------------------------------------------------------- #
# multiclass summary
# --------------------------------------------------------------------------- #
def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]
) -> dict[str, float]:
    """The four headline multiclass numbers, computed on a fixed label set."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    matrix = _confusion(y_true, y_pred, np.asarray(sorted(labels)))
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, labels=labels),
        "balanced_accuracy": _balanced_accuracy_from_confusion(matrix),
        "f1_macro": _macro_f1_from_confusion(matrix),
        "accuracy": float((y_true == y_pred).mean()),
        "mean_absolute_grade_error": float(np.abs(y_true - y_pred).mean()),
    }


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    return _balanced_accuracy_from_confusion(
        _confusion(np.asarray(y_true), np.asarray(y_pred), np.asarray(sorted(labels)))
    )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    return _macro_f1_from_confusion(
        _confusion(np.asarray(y_true), np.asarray(y_pred), np.asarray(sorted(labels)))
    )


# --------------------------------------------------------------------------- #
# continuous severity score
# --------------------------------------------------------------------------- #
def ordinal_scores(estimator, X: np.ndarray) -> np.ndarray:
    """A single continuous severity score per eye, whatever the estimator is.

    For a probabilistic classifier this is the expected grade
    ``sum_k k * P(grade = k)``, which respects the ordering of the classes and
    is far more useful than ``argmax`` for choosing a referral threshold.  For
    the ordinal regressor it is the raw regression output.
    """
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(estimator, "classes_", np.arange(proba.shape[1])), dtype=float)
        return proba @ classes
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X), dtype=float)
        if decision.ndim == 1:
            return decision
        classes = np.asarray(getattr(estimator, "classes_", np.arange(decision.shape[1])),
                             dtype=float)
        # Softmax over one-vs-rest margins, then the same expected-grade trick.
        shifted = decision - decision.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        weights /= weights.sum(axis=1, keepdims=True)
        return weights @ classes
    return np.asarray(estimator.predict(X), dtype=float)


# --------------------------------------------------------------------------- #
# referable disease
# --------------------------------------------------------------------------- #
def referable_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_grade: int = 2,
    target_sensitivity: float = 0.90,
) -> dict[str, float]:
    """Screening performance for the binary "refer / do not refer" decision.

    The operating point is not 0.5: it is the lowest score cut-off that still
    reaches ``target_sensitivity``, because a screening programme is defined by
    the fraction of sight-threatening disease it is allowed to miss.  The
    specificity at that point is the number that determines the workload of the
    ophthalmology clinic.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    positive = (y_true >= threshold_grade).astype(int)

    out = {
        "referable_prevalence": float(positive.mean()),
        "referable_auroc": float("nan"),
        "referable_ap": float("nan"),
        "referable_threshold": float("nan"),
        "referable_sensitivity": float("nan"),
        "referable_specificity": float("nan"),
        "referable_ppv": float("nan"),
        "referable_npv": float("nan"),
    }
    if positive.min() == positive.max():  # only one class present
        return out

    out["referable_auroc"] = float(roc_auc_score(positive, scores))
    out["referable_ap"] = float(average_precision_score(positive, scores))

    fpr, tpr, thresholds = roc_curve(positive, scores)
    feasible = np.flatnonzero(tpr >= target_sensitivity)
    index = int(feasible[np.argmin(fpr[feasible])]) if feasible.size else int(np.argmax(tpr))
    cutoff = float(thresholds[index])

    predicted = (scores >= cutoff).astype(int)
    tp = int(((predicted == 1) & (positive == 1)).sum())
    fp = int(((predicted == 1) & (positive == 0)).sum())
    tn = int(((predicted == 0) & (positive == 0)).sum())
    fn = int(((predicted == 0) & (positive == 1)).sum())

    out["referable_threshold"] = cutoff
    out["referable_sensitivity"] = tp / max(tp + fn, 1)
    out["referable_specificity"] = tn / max(tn + fp, 1)
    out["referable_ppv"] = tp / max(tp + fp, 1)
    out["referable_npv"] = tn / max(tn + fn, 1)
    return out


# --------------------------------------------------------------------------- #
# uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for any metric.

    A single hold-out number without an interval is not evidence; with a few
    hundred test eyes the interval is typically +-0.05 QWK, which is exactly the
    context a reader needs before believing a ranking.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    point = float(metric(y_true, y_pred))
    n = len(y_true)
    if n < 8 or n_resamples < 10:
        return point, point, point

    generator = np.random.default_rng(seed)
    if stratify:
        groups = [np.flatnonzero(y_true == value) for value in np.unique(y_true)]
    else:
        groups = [np.arange(n)]

    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = np.concatenate(
            [generator.choice(group, size=len(group), replace=True) for group in groups]
        )
        samples[i] = metric(y_true[indices], y_pred[indices])

    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lower, upper


def paired_bootstrap_test(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Is model A better than model B on the same eyes?

    Resampling the *same* indices for both models removes the between-patient
    variance that makes two independent intervals look inconclusive, which is
    why a paired test is the right tool for comparing models on one test set.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    observed = float(metric(y_true, y_pred_a)) - float(metric(y_true, y_pred_b))
    n = len(y_true)
    if n < 8:
        return {"difference": observed, "ci_low": observed, "ci_high": observed, "p_value": 1.0}

    generator = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        indices = generator.integers(0, n, size=n)
        differences[i] = float(metric(y_true[indices], y_pred_a[indices])) - float(
            metric(y_true[indices], y_pred_b[indices])
        )

    # Two-sided bootstrap p-value: how often the resampled difference lands on
    # the opposite side of zero from the observed one.
    if observed >= 0:
        p_value = float((differences <= 0).mean())
    else:
        p_value = float((differences >= 0).mean())
    p_value = min(1.0, 2.0 * p_value)

    return {
        "difference": observed,
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "p_value": p_value,
    }


def metric_table(rows: Iterable[dict], keys: Sequence[str]) -> list[dict]:
    """Helper for the report: keep only the requested keys, in order."""
    return [{key: row.get(key) for key in keys} for row in rows]
