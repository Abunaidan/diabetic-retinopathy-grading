"""Estimators and their search spaces, including an ordinal-aware classifier.

Diabetic retinopathy grades are *ordered*: predicting grade 4 for a healthy eye
is a far worse error than predicting grade 1, yet a plain multiclass classifier
treats both as "one mistake".  ``OrdinalThresholdClassifier`` fixes that by
regressing the grade as a continuous severity score and then learning the cut
points that maximise quadratic weighted kappa - the approach that dominated the
Kaggle retinopathy leaderboards and that remains the strongest simple baseline
for ordinal targets.

Every model is wrapped in the same preprocessing pipeline (median imputation,
constant-feature removal, standardisation) so that the comparison between them
is a comparison of the models, not of their preprocessing.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..utils import get_logger
from .metrics import quadratic_weighted_kappa

LOGGER = get_logger("models")

__all__ = [
    "OrdinalThresholdClassifier",
    "build_estimator",
    "list_models",
    "MODEL_DESCRIPTIONS",
]

MODEL_DESCRIPTIONS: dict[str, str] = {
    "dummy_frequent": "Always predicts the majority grade - the floor any model must clear.",
    "dummy_stratified": "Samples grades from the training prevalence - the floor for chance agreement.",
    "logreg": "Multinomial logistic regression with balanced class weights (linear, fully interpretable).",
    "random_forest": "Random forest with balanced subsampling (non-linear, robust to feature scale).",
    "hist_gbm": "Histogram gradient boosting - strong tabular baseline with class weighting.",
    "ordinal_gbm": "Gradient-boosted regression on the grade plus QWK-optimal cut points (ordinal-aware).",
}


# --------------------------------------------------------------------------- #
# ordinal classifier
# --------------------------------------------------------------------------- #
class OrdinalThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Regress the ordinal grade, then cut the score into classes.

    Parameters
    ----------
    estimator:
        Any regressor.  Defaults to histogram gradient boosting.
    cv:
        Number of internal folds used to obtain out-of-fold scores for fitting
        the cut points.  Fitting thresholds on in-sample predictions would place
        them where the training data is already separated and generalise badly,
        so the thresholds are always learned on held-out scores.
    random_state:
        Seed for the internal fold assignment.

    Attributes
    ----------
    thresholds_:
        ``n_classes - 1`` increasing cut points on the severity score.
    """

    def __init__(self, estimator: Any = None, cv: int = 5, random_state: int = 42):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state

    # -- helpers ----------------------------------------------------------- #
    def _make_estimator(self):
        if self.estimator is None:
            return HistGradientBoostingRegressor(random_state=self.random_state)
        return clone(self.estimator)

    @staticmethod
    def _apply(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        return np.searchsorted(np.asarray(thresholds, dtype=float), np.asarray(scores, dtype=float))

    def _optimise_thresholds(
        self, scores: np.ndarray, y_index: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Coordinate ascent on QWK over the cut points.

        The objective is piecewise constant, so gradient-free scalar search per
        threshold (over candidate positions between observed scores) is both
        exact within the candidate set and deterministic - unlike Nelder-Mead,
        which stalls on the flat regions of a step function.
        """
        labels = list(range(n_classes))
        quantiles = np.cumsum(np.bincount(y_index, minlength=n_classes) / len(y_index))[:-1]
        thresholds = np.quantile(scores, np.clip(quantiles, 0.0, 1.0)).astype(float)
        thresholds = np.sort(thresholds)

        order = np.sort(np.unique(scores))
        if order.size > 400:  # keep the candidate set small and evenly spread
            order = np.quantile(order, np.linspace(0.0, 1.0, 400))
        candidates = np.unique((order[:-1] + order[1:]) / 2.0) if order.size > 1 else order

        best = quadratic_weighted_kappa(y_index, self._apply(scores, thresholds), labels=labels)
        for _ in range(3):  # three sweeps are enough for convergence in practice
            improved = False
            for position in range(len(thresholds)):
                low = thresholds[position - 1] if position > 0 else -np.inf
                high = thresholds[position + 1] if position + 1 < len(thresholds) else np.inf
                window = candidates[(candidates > low) & (candidates < high)]
                for candidate in window:
                    trial = thresholds.copy()
                    trial[position] = candidate
                    score = quadratic_weighted_kappa(
                        y_index, self._apply(scores, trial), labels=labels
                    )
                    if score > best + 1e-9:
                        best, thresholds, improved = score, trial, True
            if not improved:
                break
        return np.sort(thresholds)

    # -- scikit-learn API -------------------------------------------------- #
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("OrdinalThresholdClassifier needs at least two classes.")

        y_index = np.searchsorted(self.classes_, y)
        target = self.classes_[y_index].astype(float)

        self.estimator_ = self._make_estimator()

        # Out-of-fold scores for honest threshold fitting.
        n_splits = max(2, min(int(self.cv), len(y) // 2))
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(clone(self.estimator_), X, target, cv=folds)

        self.thresholds_ = self._optimise_thresholds(np.asarray(oof, dtype=float), y_index,
                                                     n_classes)
        self.estimator_.fit(X, target)
        self.oof_qwk_ = quadratic_weighted_kappa(
            y_index, self._apply(oof, self.thresholds_), labels=list(range(n_classes))
        )
        return self

    def decision_function(self, X) -> np.ndarray:
        """Continuous severity score - the natural input to a referral cut-off."""
        check_is_fitted(self, "estimator_")
        return np.asarray(self.estimator_.predict(np.asarray(X, dtype=float)), dtype=float)

    def predict(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.clip(self._apply(scores, self.thresholds_), 0,
                                     len(self.classes_) - 1)]


# --------------------------------------------------------------------------- #
# pipelines and search spaces
# --------------------------------------------------------------------------- #
def _preprocessor() -> list[tuple[str, Any]]:
    """Shared preprocessing: impute -> drop constants -> standardise.

    Imputation is defensive (the descriptors are total, but a future feature may
    not be); the variance filter removes descriptors that are constant on the
    training fold, which would otherwise make the linear model's coefficients
    unidentifiable.
    """
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=1e-10)),
        ("scale", StandardScaler()),
    ]


def build_estimator(name: str, seed: int = 42) -> tuple[Pipeline, dict[str, Any]]:
    """Return ``(pipeline, search_space)`` for a model name."""
    steps = _preprocessor()

    if name == "dummy_frequent":
        return Pipeline(steps + [("clf", DummyClassifier(strategy="most_frequent"))]), {}

    if name == "dummy_stratified":
        return (
            Pipeline(steps + [("clf", DummyClassifier(strategy="stratified",
                                                      random_state=seed))]),
            {},
        )

    if name == "logreg":
        model = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed, solver="lbfgs"
        )
        space = {"clf__C": loguniform(1e-3, 1e2)}
        return Pipeline(steps + [("clf", model)]), space

    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        space = {
            "clf__max_depth": [None, 6, 10, 16, 24],
            "clf__min_samples_leaf": randint(1, 12),
            "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
            "clf__n_estimators": [300, 500, 800],
        }
        return Pipeline(steps + [("clf", model)]), space

    if name == "hist_gbm":
        model = HistGradientBoostingClassifier(
            random_state=seed, class_weight="balanced", early_stopping=False
        )
        space = {
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__max_leaf_nodes": randint(6, 48),
            "clf__min_samples_leaf": randint(5, 40),
            "clf__l2_regularization": loguniform(1e-4, 10.0),
            "clf__max_iter": [150, 300, 500],
        }
        return Pipeline(steps + [("clf", model)]), space

    if name == "ordinal_gbm":
        model = OrdinalThresholdClassifier(
            estimator=HistGradientBoostingRegressor(random_state=seed, early_stopping=False),
            cv=5,
            random_state=seed,
        )
        space = {
            "clf__estimator__learning_rate": loguniform(0.01, 0.3),
            "clf__estimator__max_leaf_nodes": randint(6, 48),
            "clf__estimator__min_samples_leaf": randint(5, 40),
            "clf__estimator__l2_regularization": loguniform(1e-4, 10.0),
            "clf__estimator__max_iter": [150, 300, 500],
        }
        return Pipeline(steps + [("clf", model)]), space

    raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_DESCRIPTIONS)}")


def list_models() -> list[str]:
    return list(MODEL_DESCRIPTIONS)


def uniform_space(low: float, high: float):  # small convenience re-export
    return uniform(low, high - low)


def baseline_names() -> Sequence[str]:
    return ("dummy_frequent", "dummy_stratified")
