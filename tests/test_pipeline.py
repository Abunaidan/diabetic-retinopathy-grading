"""Configuration handling and a full end-to-end run on a miniature cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dr.config import Config
from dr.features.extract import extract_dataset


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_default_config_loads_with_nested_dataclasses():
    cfg = Config.load()
    assert cfg.seed == 42
    assert cfg.features.work_size > 0
    assert isinstance(cfg.modeling.models, list) and cfg.modeling.models
    assert cfg.features.__class__.__name__ == "FeatureConfig"


def test_overrides_are_merged_not_replaced():
    cfg = Config.load(overrides={"modeling": {"cv_folds": 3}})
    assert cfg.modeling.cv_folds == 3
    assert cfg.modeling.n_bootstrap == Config().modeling.n_bootstrap


def test_unknown_keys_are_ignored_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        Config.load(overrides={"data": {"not_a_real_key": 1}})
    assert any("not_a_real_key" in record.message for record in caplog.records)


def test_paths_resolve_against_the_project_root(tmp_path):
    cfg = Config.load()
    cfg.project_root = str(tmp_path)
    assert cfg.raw_dir == tmp_path / "data" / "raw"
    assert cfg.features_path.parent == tmp_path / "data" / "processed"


def test_config_round_trips_to_a_dictionary():
    cfg = Config.load()
    payload = cfg.to_dict()
    assert json.loads(json.dumps(payload))["modeling"]["cv_folds"] == cfg.modeling.cv_folds


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_full_training_run_produces_a_complete_result_bundle(small_config, manifest):
    from dr.modeling.train import run_training

    cfg = small_config
    features = extract_dataset(manifest, cfg)
    results = run_training(cfg, features)

    # structure
    for key in ("models", "best_model", "split", "dataset", "comparison_vs_baseline",
                "leakage_audit", "permutation_importance", "ablation"):
        assert key in results, key

    # the baselines must always be present so a reader can calibrate the numbers
    assert "dummy_frequent" in results["models"]
    assert results["best_model"] not in ("dummy_frequent", "dummy_stratified")

    # every metric is finite and in range
    for name, payload in results["models"].items():
        metrics = payload["metrics"]
        assert -1.0 <= metrics["qwk"] <= 1.0, name
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0, name
        assert len(payload["y_pred"]) == results["split"]["n_test"]

    # the hold-out set is group disjoint from training
    assert results["split"]["group_overlap"] == 0

    # artefacts are on disk and reloadable
    artifacts = Path(cfg.artifacts_dir)
    assert (artifacts / "results.json").exists()
    assert (artifacts / "test_predictions.csv").exists()

    from joblib import load

    bundle = load(artifacts / "model.joblib")
    assert bundle["model_name"] == results["best_model"]
    assert len(bundle["feature_names"]) == results["dataset"]["n_features"]

    X = features[bundle["feature_names"]].to_numpy(dtype=float)
    predictions = bundle["model"].predict(X)
    assert len(predictions) == len(features)
    assert set(np.unique(predictions)).issubset(set(bundle["labels"]))


@pytest.mark.slow
def test_report_is_written_with_every_figure(small_config, manifest):
    from dr.modeling.report import build_report
    from dr.modeling.train import run_training
    from dr.utils import load_json

    cfg = small_config
    features = extract_dataset(manifest, cfg)
    results_path = Path(cfg.artifacts_dir) / "results.json"
    results = load_json(results_path) if results_path.exists() else run_training(cfg, features)

    path = build_report(cfg, features, results)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Quadratic weighted kappa" in text or "quadratic weighted kappa" in text.lower()
    assert "Limitations" in text

    figures = sorted(Path(cfg.figures_dir).glob("*.png"))
    assert len(figures) >= 6
    assert all(figure.stat().st_size > 1000 for figure in figures)


def test_config_paths_are_portable():
    """A committed config must not hard-code one machine's home directory.

    ``raw_dir: C:/Users/someone/dr-data`` is silently useless to everybody else
    who clones the repository, and nothing else in the test suite would catch it.
    """
    import re

    configs = Path(__file__).resolve().parents[1] / "configs"
    pattern = re.compile(r"^\s*\w+:\s*(?:[A-Za-z]:[/\\]|/home/|/Users/)")

    offenders = [
        f"{path.name}: {line.strip()}"
        for path in sorted(configs.glob("*.yaml"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if pattern.search(line)
    ]
    assert not offenders, "machine-specific paths in configs: " + "; ".join(offenders)


def test_tilde_paths_expand_to_the_home_directory(tmp_path):
    from pathlib import Path

    cfg = Config.load()
    cfg.project_root = str(tmp_path)
    assert cfg.resolve("~/dr-data/x") == Path.home() / "dr-data" / "x"
    assert cfg.resolve("data/raw") == tmp_path / "data" / "raw"
    absolute = Path(tmp_path / "elsewhere")
    assert cfg.resolve(absolute) == absolute


def test_every_source_file_is_tracked_by_git():
    """Guard against .gitignore silently swallowing part of the package.

    An unanchored ``data/`` rule matches ``src/dr/data/`` as well as the cohort
    directory, which once removed a whole subpackage from the published
    repository while every local test still passed.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")

    tracked = subprocess.run(
        ["git", "ls-files", "src", "tests", "configs", "scripts"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.split()
    tracked_paths = {(root / p).resolve() for p in tracked}

    expected = [
        p for p in (root / "src").rglob("*.py") if "__pycache__" not in p.parts
    ] + [p for p in (root / "tests").rglob("*.py") if "__pycache__" not in p.parts]

    missing = sorted(str(p.relative_to(root)) for p in expected if p.resolve() not in tracked_paths)
    assert not missing, "source files excluded from git: " + ", ".join(missing)
