"""Training, model selection and honest evaluation.

The protocol, in the order it is executed:

1. **Group-stratified hold-out.**  25 % of *patients* are locked away before
   anything is fitted, tuned or inspected.
2. **Baselines.**  Majority-class and prevalence-matched random classifiers.
   A model that does not beat both of them has not learned anything.
3. **Tuning.**  ``RandomizedSearchCV`` over each model's search space, scored by
   quadratic weighted kappa on a patient-grouped ``StratifiedGroupKFold`` of the
   *training* half only.
4. **Nested cross-validation.**  The whole tune-and-fit procedure is re-run
   inside an outer group CV over the full dataset, which gives a generalisation
   estimate that is not contaminated by model selection.
5. **Hold-out evaluation.**  Bootstrap confidence intervals, a paired bootstrap
   against the baseline, and the referable-DR operating point.
6. **Leakage audit.**  The same model is cross-validated once *ignoring* the
   patient grouping, to quantify how much a naive random split would have
   inflated the result.
7. **Explanation.**  Permutation importance on held-out data, aggregated per
   descriptor family, plus a per-family ablation study.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from ..config import GRADE_NAMES, Config
from ..features.extract import feature_group, select_feature_columns
from ..utils import ensure_dir, environment_fingerprint, get_logger, save_json, timer
from .metrics import (
    balanced_accuracy,
    bootstrap_ci,
    classification_metrics,
    macro_f1,
    ordinal_scores,
    paired_bootstrap_test,
    quadratic_weighted_kappa,
    referable_metrics,
)
from .models import MODEL_DESCRIPTIONS, build_estimator
from .splits import group_stratified_holdout, make_cv, safe_n_splits, split_summary

LOGGER = get_logger("train")

BASELINES = ("dummy_frequent", "dummy_stratified")


# --------------------------------------------------------------------------- #
# data preparation
# --------------------------------------------------------------------------- #
def prepare_matrix(features: pd.DataFrame) -> dict[str, Any]:
    """Split the feature table into X, y, groups and column names."""
    columns = select_feature_columns(features)
    if not columns:
        raise ValueError("No numeric feature columns found - run `dr features` first.")

    X = features[columns].to_numpy(dtype=float)
    y = features["grade"].to_numpy(dtype=int)
    groups = features["group"].astype(str).to_numpy()

    # Constant descriptors carry no information and only widen the search space.
    keep = X.std(axis=0) > 1e-12
    if not keep.all():
        dropped = [c for c, k in zip(columns, keep) if not k]
        LOGGER.info("Dropping %d constant descriptors: %s", len(dropped), dropped[:6])
        X = X[:, keep]
        columns = [c for c, k in zip(columns, keep) if k]

    labels = sorted(np.unique(y).tolist())
    LOGGER.info(
        "Design matrix: %d eyes x %d descriptors | %d patients | classes %s",
        X.shape[0], X.shape[1], len(np.unique(groups)), labels,
    )
    return {"X": X, "y": y, "groups": groups, "columns": columns, "labels": labels}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _qwk_scorer_for(labels: list[int]):
    from sklearn.metrics import make_scorer

    return make_scorer(
        lambda y_true, y_pred: quadratic_weighted_kappa(y_true, y_pred, labels=labels)
    )


def _fit_with_search(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[int],
    cfg: Config,
    n_iter: int | None = None,
    seed_offset: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Tune one model on the given data and return the refitted best estimator."""
    seed = cfg.seed + seed_offset
    estimator, space = build_estimator(name, seed=seed)
    scorer = _qwk_scorer_for(labels)
    n_splits = safe_n_splits(y, groups, cfg.modeling.cv_folds)
    cv = make_cv(n_splits, seed, grouped=True)

    if not space:
        scores = cross_val_score(
            clone(estimator), X, y, groups=groups, cv=cv, scoring=scorer, n_jobs=-1
        )
        fitted = clone(estimator).fit(X, y)
        return fitted, {
            "best_params": {},
            "cv_qwk_mean": float(np.mean(scores)),
            "cv_qwk_std": float(np.std(scores)),
            "n_candidates": 1,
        }

    search = RandomizedSearchCV(
        estimator,
        param_distributions=space,
        n_iter=int(n_iter or cfg.modeling.n_search_iter),
        scoring=scorer,
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score=0.0,
    )
    search.fit(X, y, groups=groups)
    index = int(search.best_index_)
    return search.best_estimator_, {
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "cv_qwk_mean": float(search.cv_results_["mean_test_score"][index]),
        "cv_qwk_std": float(search.cv_results_["std_test_score"][index]),
        "n_candidates": int(len(search.cv_results_["mean_test_score"])),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _evaluate(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[int],
    cfg: Config,
) -> dict[str, Any]:
    """Hold-out metrics with bootstrap intervals and the referral operating point."""
    y_pred = np.asarray(estimator.predict(X_test)).astype(int)
    scores = ordinal_scores(estimator, X_test)

    metrics = classification_metrics(y_test, y_pred, labels=labels)
    intervals: dict[str, list[float]] = {}
    for key, function in (
        ("qwk", lambda a, b: quadratic_weighted_kappa(a, b, labels=labels)),
        ("balanced_accuracy", lambda a, b: balanced_accuracy(a, b, labels)),
        ("f1_macro", lambda a, b: macro_f1(a, b, labels)),
    ):
        point, low, high = bootstrap_ci(
            function, y_test, y_pred, n_resamples=cfg.modeling.n_bootstrap, seed=cfg.seed
        )
        intervals[key] = [point, low, high]

    metrics.update(
        referable_metrics(
            y_test,
            scores,
            threshold_grade=cfg.modeling.referable_threshold_grade,
            target_sensitivity=cfg.modeling.target_sensitivity,
        )
    )
    return {
        "metrics": metrics,
        "confidence_intervals": intervals,
        "y_pred": y_pred.tolist(),
        "scores": np.asarray(scores, dtype=float).tolist(),
    }


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def run_training(cfg: Config, features: pd.DataFrame) -> dict[str, Any]:
    """Execute the full protocol and return a JSON-serialisable result bundle."""
    data = prepare_matrix(features)
    X, y, groups, columns, labels = (
        data["X"], data["y"], data["groups"], data["columns"], data["labels"]
    )

    started = time.perf_counter()
    train_index, test_index = group_stratified_holdout(
        y, groups, test_size=cfg.modeling.test_size, seed=cfg.seed
    )
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    groups_train, groups_test = groups[train_index], groups[test_index]

    split_info = split_summary(y_train, y_test, groups_train, groups_test, labels)
    LOGGER.info(
        "Hold-out: %d train eyes (%d patients) / %d test eyes (%d patients)",
        split_info["n_train"], split_info["n_train_groups"],
        split_info["n_test"], split_info["n_test_groups"],
    )

    results: dict[str, Any] = {
        "environment": environment_fingerprint(),
        "config": cfg.to_dict(),
        "dataset": {
            "n_eyes": int(len(y)),
            "n_patients": int(pd.unique(groups).size),
            "n_features": int(X.shape[1]),
            "feature_names": columns,
            "class_counts": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
            "grade_names": {int(k): v for k, v in GRADE_NAMES.items() if k in labels},
        },
        "split": split_info,
        "models": {},
    }

    model_names = list(BASELINES) + [m for m in cfg.modeling.models if m not in BASELINES]
    fitted_models: dict[str, Any] = {}

    for name in model_names:
        with timer(f"Model '{name}'", LOGGER):
            estimator, search_info = _fit_with_search(
                name, X_train, y_train, groups_train, labels, cfg
            )
            evaluation = _evaluate(estimator, X_test, y_test, labels, cfg)
            fitted_models[name] = estimator
            results["models"][name] = {
                "description": MODEL_DESCRIPTIONS.get(name, ""),
                **search_info,
                **evaluation,
            }
            LOGGER.info(
                "%-16s CV QWK %.3f +- %.3f | test QWK %.3f | balanced acc %.3f",
                name,
                search_info["cv_qwk_mean"],
                search_info["cv_qwk_std"],
                evaluation["metrics"]["qwk"],
                evaluation["metrics"]["balanced_accuracy"],
            )

    # ---- model selection is made on cross-validation, never on the test set --
    candidates = {k: v for k, v in results["models"].items() if k not in BASELINES}
    best_name = max(candidates, key=lambda k: candidates[k]["cv_qwk_mean"])
    results["best_model"] = best_name
    results["selection_rule"] = (
        "highest patient-grouped cross-validated QWK on the training half; "
        "the hold-out set was not consulted for selection"
    )
    LOGGER.info("Selected '%s' on cross-validated QWK", best_name)

    # ---- is the winner actually better than the baseline? -------------------
    baseline_pred = np.asarray(results["models"]["dummy_frequent"]["y_pred"])
    best_pred = np.asarray(results["models"][best_name]["y_pred"])
    results["comparison_vs_baseline"] = paired_bootstrap_test(
        lambda a, b: quadratic_weighted_kappa(a, b, labels=labels),
        y_test,
        best_pred,
        baseline_pred,
        n_resamples=cfg.modeling.n_bootstrap,
        seed=cfg.seed,
    )

    runner_up = sorted(
        (k for k in candidates if k != best_name),
        key=lambda k: candidates[k]["cv_qwk_mean"],
        reverse=True,
    )
    if runner_up:
        results["comparison_vs_runner_up"] = {
            "runner_up": runner_up[0],
            **paired_bootstrap_test(
                lambda a, b: quadratic_weighted_kappa(a, b, labels=labels),
                y_test,
                best_pred,
                np.asarray(results["models"][runner_up[0]]["y_pred"]),
                n_resamples=cfg.modeling.n_bootstrap,
                seed=cfg.seed,
            ),
        }

    # ---- nested cross-validation -------------------------------------------
    if cfg.modeling.nested_cv:
        with timer("Nested cross-validation", LOGGER):
            results["nested_cv"] = _nested_cv(cfg, X, y, groups, labels, candidates.keys())

    # ---- leakage audit ------------------------------------------------------
    with timer("Leakage audit (grouped vs random CV)", LOGGER):
        results["leakage_audit"] = _leakage_audit(cfg, X, y, groups, labels, best_name)

    # ---- explanation --------------------------------------------------------
    with timer("Permutation importance", LOGGER):
        results["permutation_importance"] = _permutation_importance(
            fitted_models[best_name], X_test, y_test, columns, labels, cfg
        )

    with timer("Feature-family ablation", LOGGER):
        results["ablation"] = _family_ablation(
            cfg, X_train, y_train, groups_train, columns, labels, best_name
        )

    # ---- persistence --------------------------------------------------------
    results["test_truth"] = y_test.tolist()
    results["test_image_ids"] = features.iloc[test_index]["image_id"].astype(str).tolist()
    results["test_groups"] = groups_test.tolist()
    results["runtime_seconds"] = float(time.perf_counter() - started)

    artifacts = ensure_dir(cfg.artifacts_dir)
    dump(
        {
            "model": fitted_models[best_name],
            "model_name": best_name,
            "feature_names": columns,
            "labels": labels,
            "config": cfg.to_dict(),
            "referable_threshold": results["models"][best_name]["metrics"].get(
                "referable_threshold"
            ),
        },
        artifacts / "model.joblib",
    )
    save_json(results, artifacts / "results.json")
    _save_predictions(cfg, features, test_index, results)
    LOGGER.info("Artifacts written to %s", artifacts)
    return results


# --------------------------------------------------------------------------- #
# protocol components
# --------------------------------------------------------------------------- #
def _nested_cv(
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[int],
    names,
) -> dict[str, Any]:
    """Outer group CV around the whole tuning procedure.

    A single hold-out number depends on which patients happened to land in the
    test set; nested CV re-runs selection inside every outer fold, so the spread
    across folds is an honest estimate of how the *procedure* generalises.
    """
    n_outer = safe_n_splits(y, groups, cfg.modeling.nested_outer_folds)
    outer = make_cv(n_outer, cfg.seed + 1000, grouped=True)
    inner_iter = max(8, cfg.modeling.n_search_iter // 3)

    out: dict[str, Any] = {"n_outer_folds": n_outer, "inner_search_iter": inner_iter,
                           "models": {}}
    for name in names:
        fold_scores: list[float] = []
        for fold, (train_index, test_index) in enumerate(outer.split(X, y, groups)):
            estimator, _ = _fit_with_search(
                name,
                X[train_index],
                y[train_index],
                groups[train_index],
                labels,
                cfg,
                n_iter=inner_iter,
                seed_offset=100 + fold,
            )
            prediction = estimator.predict(X[test_index])
            fold_scores.append(quadratic_weighted_kappa(y[test_index], prediction, labels=labels))
        out["models"][name] = {
            "fold_scores": [float(s) for s in fold_scores],
            "mean": float(np.mean(fold_scores)),
            "std": float(np.std(fold_scores)),
        }
        LOGGER.info(
            "Nested CV %-16s QWK %.3f +- %.3f", name, np.mean(fold_scores), np.std(fold_scores)
        )
    return out


def _leakage_audit(
    cfg: Config,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[int],
    model_name: str,
) -> dict[str, Any]:
    """Quantify the optimism of a naive random split on this exact dataset.

    Both eyes of a patient share camera, illumination and disease state.  If
    they are allowed on both sides of the split, the model can recognise the
    patient rather than the pathology.  Reporting the size of that gap is more
    convincing than merely claiming the split is correct.
    """
    estimator, _ = build_estimator(model_name, seed=cfg.seed)
    scorer = _qwk_scorer_for(labels)
    n_splits = safe_n_splits(y, groups, cfg.modeling.cv_folds)

    grouped = cross_val_score(
        clone(estimator), X, y, groups=groups,
        cv=make_cv(n_splits, cfg.seed, grouped=True), scoring=scorer, n_jobs=-1,
    )
    random = cross_val_score(
        clone(estimator), X, y,
        cv=make_cv(n_splits, cfg.seed, grouped=False), scoring=scorer, n_jobs=-1,
    )
    audit = {
        "model": model_name,
        "grouped_cv_qwk_mean": float(grouped.mean()),
        "grouped_cv_qwk_std": float(grouped.std()),
        "random_cv_qwk_mean": float(random.mean()),
        "random_cv_qwk_std": float(random.std()),
        "optimism": float(random.mean() - grouped.mean()),
    }
    LOGGER.info(
        "Leakage audit: random-split CV QWK %.3f vs patient-grouped %.3f (optimism %+.3f)",
        audit["random_cv_qwk_mean"], audit["grouped_cv_qwk_mean"], audit["optimism"],
    )
    return audit


def _permutation_importance(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    columns: list[str],
    labels: list[int],
    cfg: Config,
) -> dict[str, Any]:
    """Model-agnostic importance measured on held-out data.

    Impurity-based importances are biased towards high-cardinality features and
    are computed on the training set; permutation importance answers the
    question that actually matters - how much held-out QWK is lost when this
    descriptor is destroyed.
    """
    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring=_qwk_scorer_for(labels),
        n_repeats=int(cfg.modeling.permutation_repeats),
        random_state=cfg.seed,
        n_jobs=-1,
    )
    per_feature = [
        {
            "feature": column,
            "family": feature_group(column),
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for column, mean, std in zip(columns, result.importances_mean, result.importances_std)
    ]
    per_feature.sort(key=lambda row: row["importance_mean"], reverse=True)

    families: dict[str, float] = {}
    for row in per_feature:
        families[row["family"]] = families.get(row["family"], 0.0) + max(
            row["importance_mean"], 0.0
        )
    return {
        "per_feature": per_feature,
        "per_family": dict(sorted(families.items(), key=lambda kv: kv[1], reverse=True)),
        "n_repeats": int(cfg.modeling.permutation_repeats),
        "scoring": "quadratic weighted kappa",
    }


def _family_ablation(
    cfg: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    columns: list[str],
    labels: list[int],
    model_name: str,
) -> dict[str, Any]:
    """Cross-validated score using one descriptor family at a time, and without it.

    Permutation importance ranks individual descriptors; correlated families
    (the four lesion channels, say) share their importance and each looks weak.
    Ablating whole families answers the question a clinician would ask: does the
    lesion evidence carry the signal, or is it all texture?
    """
    scorer = _qwk_scorer_for(labels)
    n_splits = safe_n_splits(y_train, groups_train, cfg.modeling.cv_folds)
    cv = make_cv(n_splits, cfg.seed, grouped=True)
    estimator, _ = build_estimator(model_name, seed=cfg.seed)

    families = sorted({feature_group(c) for c in columns})
    index = {family: [i for i, c in enumerate(columns) if feature_group(c) == family]
             for family in families}

    def score(subset: list[int]) -> float:
        if not subset:
            return 0.0
        values = cross_val_score(
            clone(estimator), X_train[:, subset], y_train, groups=groups_train,
            cv=cv, scoring=scorer, n_jobs=-1,
        )
        return float(values.mean())

    full = score(list(range(len(columns))))
    rows = []
    for family in families:
        only = index[family]
        without = [i for i in range(len(columns)) if i not in set(only)]
        qwk_without = score(without)
        rows.append(
            {
                "family": family,
                "n_features": len(only),
                "qwk_family_only": score(only),
                "qwk_without_family": qwk_without,
                "delta_when_removed": qwk_without - full,
            }
        )
    rows.sort(key=lambda r: r["qwk_family_only"], reverse=True)
    return {"model": model_name, "qwk_all_features": full, "families": rows}


def _save_predictions(
    cfg: Config, features: pd.DataFrame, test_index: np.ndarray, results: dict[str, Any]
) -> Path:
    """Per-eye predictions of every model on the hold-out set, for error analysis."""
    frame = pd.DataFrame(
        {
            "image_id": features.iloc[test_index]["image_id"].astype(str).to_numpy(),
            "group": features.iloc[test_index]["group"].astype(str).to_numpy(),
            "grade_true": np.asarray(results["test_truth"], dtype=int),
        }
    )
    for name, payload in results["models"].items():
        frame[f"pred_{name}"] = np.asarray(payload["y_pred"], dtype=int)
        frame[f"score_{name}"] = np.asarray(payload["scores"], dtype=float)

    path = ensure_dir(cfg.artifacts_dir) / "test_predictions.csv"
    frame.to_csv(path, index=False)
    return path
