"""The ordinal classifier must behave like any other scikit-learn estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dr.modeling.metrics import ordinal_scores, quadratic_weighted_kappa
from dr.modeling.models import MODEL_DESCRIPTIONS, OrdinalThresholdClassifier, build_estimator


def _ordinal_problem(n: int = 240, seed: int = 0):
    """A latent-severity problem: exactly the structure of a retinopathy grade."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    latent = X[:, 0] * 1.6 + X[:, 1] * 0.9 + rng.normal(0, 0.5, size=n)
    y = np.digitize(latent, np.quantile(latent, [0.45, 0.65, 0.85, 0.95]))
    return X, y.astype(int)


def test_fits_predicts_and_keeps_thresholds_sorted():
    X, y = _ordinal_problem()
    model = OrdinalThresholdClassifier(random_state=0).fit(X, y)
    assert list(model.classes_) == sorted(np.unique(y).tolist())
    assert len(model.thresholds_) == len(model.classes_) - 1
    assert np.all(np.diff(model.thresholds_) >= 0)

    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert set(predictions).issubset(set(model.classes_))


def test_threshold_search_beats_plain_rounding_on_the_scores_it_is_fitted_on():
    """The optimiser's contract: on its own data it never loses to rounding.

    Rounding a regression output at .5 is the implicit alternative; it ignores
    the class prevalence entirely, which is precisely what hurts on a cohort
    where 5 % of eyes carry the top grade.
    """
    rng = np.random.default_rng(0)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = rng.choice([0, 1, 2, 3, 4], size=500, p=[0.49, 0.10, 0.27, 0.08, 0.06])
        scores = y * 0.7 + rng.normal(0, 0.8, size=500)  # compressed, shifted scores
        labels = [0, 1, 2, 3, 4]

        model = OrdinalThresholdClassifier()
        thresholds = model._optimise_thresholds(scores, y, n_classes=5)
        tuned = quadratic_weighted_kappa(y, model._apply(scores, thresholds), labels=labels)
        naive = quadratic_weighted_kappa(
            y, np.clip(np.rint(scores), 0, 4).astype(int), labels=labels
        )
        assert np.all(np.diff(thresholds) >= 0)
        assert tuned >= naive


def test_is_at_least_competitive_with_naive_rounding_out_of_sample():
    """Averaged over resamples, the fitted cut points must not cost performance."""
    tuned_scores, naive_scores = [], []
    for seed in range(4):
        X, y = _ordinal_problem(n=600, seed=seed)
        labels = sorted(np.unique(y).tolist())
        X_train, X_test, y_train, y_test = X[:400], X[400:], y[:400], y[400:]

        model = OrdinalThresholdClassifier(random_state=0).fit(X_train, y_train)
        tuned_scores.append(
            quadratic_weighted_kappa(y_test, model.predict(X_test), labels=labels)
        )

        regressor = HistGradientBoostingRegressor(random_state=0).fit(
            X_train, y_train.astype(float)
        )
        rounded = np.clip(
            np.rint(regressor.predict(X_test)), min(labels), max(labels)
        ).astype(int)
        naive_scores.append(quadratic_weighted_kappa(y_test, rounded, labels=labels))

    assert np.mean(tuned_scores) >= np.mean(naive_scores) - 0.02


def test_decision_function_is_monotone_in_severity():
    X, y = _ordinal_problem(seed=5)
    model = OrdinalThresholdClassifier(random_state=0).fit(X, y)
    scores = model.decision_function(X)
    means = [scores[y == grade].mean() for grade in sorted(np.unique(y))]
    assert all(a < b for a, b in zip(means, means[1:]))


def test_ordinal_scores_uses_the_decision_function():
    X, y = _ordinal_problem(seed=6)
    model = OrdinalThresholdClassifier(random_state=0).fit(X, y)
    assert np.allclose(ordinal_scores(model, X), model.decision_function(X))


def test_is_clonable_and_exposes_nested_parameters():
    model = OrdinalThresholdClassifier(
        estimator=HistGradientBoostingRegressor(max_iter=17), cv=3, random_state=1
    )
    params = model.get_params(deep=True)
    assert params["cv"] == 3
    assert params["estimator__max_iter"] == 17

    twin = clone(model)
    assert twin.get_params(deep=True)["estimator__max_iter"] == 17


def test_works_inside_a_pipeline_and_a_grid_search():
    X, y = _ordinal_problem(n=180, seed=7)
    pipeline, _ = build_estimator("ordinal_gbm", seed=0)
    search = GridSearchCV(
        pipeline,
        {"clf__estimator__max_iter": [30, 60]},
        cv=3,
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(X, y)
    assert isinstance(search.best_estimator_, Pipeline)
    assert search.best_estimator_.predict(X).shape == y.shape


def test_thresholds_are_fitted_out_of_fold():
    """In-sample thresholds would overfit; the estimator stores its OOF score."""
    X, y = _ordinal_problem(seed=8)
    model = OrdinalThresholdClassifier(random_state=0).fit(X, y)
    assert hasattr(model, "oof_qwk_")
    assert -1.0 <= model.oof_qwk_ <= 1.0


def test_single_class_input_is_rejected():
    X = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValueError, match="at least two classes"):
        OrdinalThresholdClassifier().fit(X, np.zeros(20, dtype=int))


@pytest.mark.parametrize("name", sorted(MODEL_DESCRIPTIONS))
def test_every_registered_model_fits_and_predicts(name):
    X, y = _ordinal_problem(n=120, seed=9)
    estimator, space = build_estimator(name, seed=0)
    estimator.fit(X, y)
    predictions = estimator.predict(X)
    assert predictions.shape == y.shape
    assert set(predictions).issubset(set(np.unique(y)))
    assert isinstance(space, dict)
    assert all(key.startswith("clf__") for key in space)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_estimator("definitely_not_a_model")
