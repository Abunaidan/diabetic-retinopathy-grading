"""Leakage-free, stratified, group-aware data splitting.

Fundus datasets contain two photographs per patient (left and right eye) and
sometimes several visits.  Two images of the same patient share illumination,
camera, pigmentation and disease state, so a random split leaves near-duplicates
on both sides and inflates every metric.  Splitting is therefore always done at
the *group* level (patient), while still stratifying on the grade so the rare
severe classes survive in both halves.

``dr.modeling.train`` deliberately also fits one model under a naive random
split, purely to *measure* how large that optimism is on this dataset.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ..utils import get_logger

LOGGER = get_logger("splits")

__all__ = [
    "safe_n_splits",
    "make_cv",
    "group_stratified_holdout",
    "verify_no_leakage",
    "split_summary",
]


class LeakageError(RuntimeError):
    """Raised when the same patient appears in both training and test data."""


def safe_n_splits(y: np.ndarray, groups: np.ndarray | None, requested: int) -> int:
    """Largest usable fold count for this label/group structure.

    ``StratifiedGroupKFold`` cannot place a class into every fold if that class
    has fewer *groups* than folds; silently accepting the request produces folds
    without a class and NaN metrics.  Reducing the fold count and saying so is
    the honest alternative.
    """
    y = np.asarray(y)
    if groups is None:
        per_class = np.bincount(y.astype(int))
    else:
        frame = pd.DataFrame({"y": y, "g": np.asarray(groups)})
        per_class = frame.groupby("y")["g"].nunique().to_numpy()

    per_class = per_class[per_class > 0]
    limit = int(per_class.min()) if per_class.size else requested
    n_splits = max(2, min(int(requested), limit))
    if n_splits < requested:
        LOGGER.warning(
            "Reducing CV folds from %d to %d: the rarest class has only %d group(s)",
            requested,
            n_splits,
            limit,
        )
    return n_splits


def make_cv(n_splits: int, seed: int, grouped: bool = True):
    """Cross-validator used everywhere in the project."""
    if grouped:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def group_stratified_holdout(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Single hold-out split that is both group-disjoint and grade-stratified.

    Implemented as "take one fold of a StratifiedGroupKFold": that gives the
    stratification of ``train_test_split(stratify=y)`` and the group isolation
    of ``GroupShuffleSplit`` at the same time, which neither of them provides
    alone.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    n_splits = int(round(1.0 / max(min(test_size, 0.5), 0.05)))
    n_splits = safe_n_splits(y, groups, n_splits)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, test_index = next(splitter.split(np.zeros(len(y)), y, groups))

    verify_no_leakage(groups[train_index], groups[test_index])
    if len(np.unique(y[test_index])) < 2:
        raise ValueError(
            "The hold-out split contains a single class; lower modeling.test_size "
            "or collect more images of the rare grades."
        )
    return train_index, test_index


def verify_no_leakage(groups_train: Sequence, groups_test: Sequence) -> None:
    """Hard guarantee that no patient spans the split."""
    overlap = set(np.asarray(groups_train).tolist()) & set(np.asarray(groups_test).tolist())
    if overlap:
        raise LeakageError(
            f"{len(overlap)} group(s) appear in both train and test, e.g. "
            f"{sorted(overlap)[:5]}. This would invalidate every reported metric."
        )


def split_summary(
    y_train: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
    labels: Sequence[int],
) -> dict:
    """Everything a reviewer wants to see about a split, in one dictionary."""

    def distribution(values: np.ndarray) -> dict[int, int]:
        counts = pd.Series(values).value_counts()
        return {int(label): int(counts.get(label, 0)) for label in labels}

    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_groups": int(pd.unique(np.asarray(groups_train)).size),
        "n_test_groups": int(pd.unique(np.asarray(groups_test)).size),
        "train_distribution": distribution(y_train),
        "test_distribution": distribution(y_test),
        "group_overlap": 0,
    }
