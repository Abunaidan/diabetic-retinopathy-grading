"""Splitting is the single easiest way to fake a good result; guard it hard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dr.modeling.splits import (
    LeakageError,
    group_stratified_holdout,
    make_cv,
    safe_n_splits,
    split_summary,
    verify_no_leakage,
)


def _cohort(n_patients: int = 60, seed: int = 0):
    """Two eyes per patient, correlated grades - the real data structure."""
    rng = np.random.default_rng(seed)
    grades, groups = [], []
    for patient in range(n_patients):
        base = int(rng.integers(0, 5))
        for eye in ("left", "right"):
            grades.append(int(np.clip(base + rng.integers(-1, 2), 0, 4)))
            groups.append(f"P{patient:03d}")
    return np.array(grades), np.array(groups)


def test_holdout_never_splits_a_patient():
    y, groups = _cohort()
    train_index, test_index = group_stratified_holdout(y, groups, test_size=0.25, seed=42)
    assert set(groups[train_index]).isdisjoint(set(groups[test_index]))
    assert len(train_index) + len(test_index) == len(y)


def test_holdout_keeps_every_class_on_both_sides():
    y, groups = _cohort()
    train_index, test_index = group_stratified_holdout(y, groups, test_size=0.25, seed=42)
    assert set(np.unique(y[test_index])) == set(np.unique(y))
    assert set(np.unique(y[train_index])) == set(np.unique(y))


def test_holdout_size_is_approximately_requested():
    y, groups = _cohort(n_patients=100)
    _, test_index = group_stratified_holdout(y, groups, test_size=0.25, seed=1)
    assert 0.15 < len(test_index) / len(y) < 0.35


def test_verify_no_leakage_raises_on_overlap():
    with pytest.raises(LeakageError):
        verify_no_leakage(["a", "b"], ["b", "c"])
    verify_no_leakage(["a", "b"], ["c"])  # must not raise


def test_cv_folds_are_group_disjoint():
    y, groups = _cohort()
    cv = make_cv(4, seed=0, grouped=True)
    for train_index, test_index in cv.split(np.zeros(len(y)), y, groups):
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))


def test_safe_n_splits_backs_off_for_rare_classes():
    y = np.array([0] * 20 + [1] * 20 + [2, 2])
    groups = np.array([f"g{i}" for i in range(len(y))])
    assert safe_n_splits(y, groups, requested=5) == 2
    assert safe_n_splits(y[:40], groups[:40], requested=5) == 5


def test_split_summary_reports_distributions():
    y, groups = _cohort(n_patients=40)
    train_index, test_index = group_stratified_holdout(y, groups, seed=3)
    summary = split_summary(
        y[train_index], y[test_index], groups[train_index], groups[test_index],
        labels=sorted(np.unique(y).tolist()),
    )
    assert summary["n_train"] + summary["n_test"] == len(y)
    assert summary["n_train_groups"] + summary["n_test_groups"] == len(pd.unique(groups))
    assert sum(summary["test_distribution"].values()) == summary["n_test"]
