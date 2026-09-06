"""The metric layer is where a silent bug would invalidate every conclusion."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from dr.modeling.metrics import (
    balanced_accuracy,
    bootstrap_ci,
    classification_metrics,
    macro_f1,
    ordinal_scores,
    paired_bootstrap_test,
    quadratic_weighted_kappa,
    referable_metrics,
)

LABELS = [0, 1, 2, 3, 4]


def test_qwk_matches_scikit_learn_on_random_data():
    """Our implementation must agree with the reference to floating-point noise."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        y_true = rng.integers(0, 5, size=120)
        y_pred = rng.integers(0, 5, size=120)
        expected = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=LABELS)
        assert quadratic_weighted_kappa(y_true, y_pred, labels=LABELS) == pytest.approx(
            expected, abs=1e-9
        )


def test_fast_balanced_accuracy_and_macro_f1_match_scikit_learn():
    """The bootstrap needs fast metrics; they must stay identical to the reference."""
    rng = np.random.default_rng(1)
    for _ in range(30):
        y_true = rng.integers(0, 5, size=90)
        y_pred = rng.integers(0, 5, size=90)
        assert balanced_accuracy(y_true, y_pred, LABELS) == pytest.approx(
            balanced_accuracy_score(y_true, y_pred), abs=1e-12
        )
        assert macro_f1(y_true, y_pred, LABELS) == pytest.approx(
            f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0),
            abs=1e-12,
        )


def test_fast_metrics_match_when_a_class_is_absent():
    """The realistic degenerate case: a rare grade missing from a bootstrap fold."""
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert balanced_accuracy(y_true, y_pred, LABELS) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred), abs=1e-12
    )
    assert macro_f1(y_true, y_pred, LABELS) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0), abs=1e-12
    )


def test_qwk_perfect_and_ordering():
    y = np.array([0, 1, 2, 3, 4, 4, 2, 1])
    assert quadratic_weighted_kappa(y, y, labels=LABELS) == pytest.approx(1.0)

    near = np.clip(y + 1, 0, 4)
    far = np.abs(4 - y)
    # A one-grade error must cost less than a maximal error.
    assert quadratic_weighted_kappa(y, near, labels=LABELS) > quadratic_weighted_kappa(
        y, far, labels=LABELS
    )


def test_qwk_is_zero_not_nan_for_a_constant_predictor():
    """The majority-class baseline makes kappa undefined; 0.0 keeps CV usable."""
    y_true = np.array([0, 0, 1, 2, 3])
    y_pred = np.zeros_like(y_true)
    value = quadratic_weighted_kappa(y_true, y_pred, labels=LABELS)
    assert value == 0.0
    assert np.isfinite(value)

    assert quadratic_weighted_kappa(np.zeros(5, int), np.zeros(5, int), labels=LABELS) == 0.0


def test_classification_metrics_keys_and_ranges():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 3, 3, 0, 2, 2])
    metrics = classification_metrics(y_true, y_pred, labels=LABELS)
    for key in ("qwk", "balanced_accuracy", "f1_macro", "accuracy", "mean_absolute_grade_error"):
        assert key in metrics and np.isfinite(metrics[key])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["mean_absolute_grade_error"] == pytest.approx(2 / 8)


def test_referable_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 3.0, 3.5, 3.9, 4.0])
    out = referable_metrics(y_true, scores, threshold_grade=2, target_sensitivity=0.9)
    assert out["referable_auroc"] == pytest.approx(1.0)
    assert out["referable_sensitivity"] == pytest.approx(1.0)
    assert out["referable_specificity"] == pytest.approx(1.0)
    assert out["referable_prevalence"] == pytest.approx(0.5)


def test_referable_metrics_single_class_is_reported_as_nan_not_crash():
    y_true = np.zeros(10, dtype=int)
    out = referable_metrics(y_true, np.linspace(0, 1, 10))
    assert np.isnan(out["referable_auroc"])


def test_referable_operating_point_respects_target_sensitivity():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, size=400)
    scores = y_true + rng.normal(0, 1.0, size=400)
    out = referable_metrics(y_true, scores, threshold_grade=2, target_sensitivity=0.9)
    assert out["referable_sensitivity"] >= 0.9 - 1e-9


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, size=200)
    y_pred = np.clip(y_true + rng.integers(-1, 2, size=200), 0, 4)
    point, low, high = bootstrap_ci(
        lambda a, b: quadratic_weighted_kappa(a, b, labels=LABELS),
        y_true, y_pred, n_resamples=200, seed=0,
    )
    assert low <= point <= high
    assert high - low > 0


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 5, size=300)
    good = y_true.copy()
    bad = rng.integers(0, 5, size=300)
    out = paired_bootstrap_test(
        lambda a, b: quadratic_weighted_kappa(a, b, labels=LABELS),
        y_true, good, bad, n_resamples=200, seed=0,
    )
    assert out["difference"] > 0.5
    assert out["p_value"] < 0.05
    assert out["ci_low"] <= out["difference"] <= out["ci_high"]


def test_ordinal_scores_uses_expected_grade_for_probabilistic_models():
    class FakeProbabilistic:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, X):
            return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    scores = ordinal_scores(FakeProbabilistic(), np.zeros((3, 2)))
    assert scores.tolist() == [0.0, 2.0, 1.0]


def test_ordinal_scores_falls_back_to_decision_function():
    class FakeOrdinal:
        def decision_function(self, X):
            return np.array([0.5, 3.2])

    assert ordinal_scores(FakeOrdinal(), np.zeros((2, 2))).tolist() == [0.5, 3.2]
