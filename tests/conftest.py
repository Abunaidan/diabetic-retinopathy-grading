"""Shared fixtures: a miniature synthetic cohort built once per test session."""

from __future__ import annotations

from pathlib import Path

import pytest

from dr.config import Config


@pytest.fixture(scope="session")
def small_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """A configuration pointing at a throwaway directory, sized for speed."""
    root = tmp_path_factory.mktemp("dr_project")
    cfg = Config.load()
    cfg.project_root = str(root)
    cfg.seed = 7
    cfg.paths.raw_dir = "data/raw"
    cfg.paths.labels_csv = None
    cfg.data.dataset = "synthetic"
    cfg.synthetic.n_patients = 8
    cfg.synthetic.image_size = 256
    cfg.features.work_size = 192
    cfg.features.n_jobs = 1
    cfg.features.cache = False
    cfg.modeling.cv_folds = 3
    cfg.modeling.n_search_iter = 2
    cfg.modeling.n_bootstrap = 50
    cfg.modeling.nested_cv = False
    cfg.modeling.permutation_repeats = 2
    cfg.modeling.models = ["logreg"]
    return cfg


@pytest.fixture(scope="session")
def synthetic_dataset(small_config: Config) -> tuple[Path, Path]:
    from dr.data.synthetic import generate_dataset

    return generate_dataset(small_config)


@pytest.fixture(scope="session")
def manifest(small_config: Config, synthetic_dataset):
    from dr.data.datasets import build_manifest

    frame, _meta = build_manifest(small_config)
    return frame


@pytest.fixture(scope="session")
def sample_image(manifest) -> str:
    return str(manifest.iloc[0]["image_path"])
