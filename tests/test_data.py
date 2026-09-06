"""Label joining - the failure mode this project exists to prevent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dr.config import Config
from dr.data.datasets import (
    ADAPTERS,
    LabelJoinError,
    build_manifest,
    derive_groups,
    discover_images,
    load_labels,
    subsample_by_patient,
    summarise_manifest,
)


def test_synthetic_cohort_is_written_with_a_labels_csv(small_config, synthetic_dataset):
    image_dir, csv_path = synthetic_dataset
    assert csv_path.exists()
    frame = pd.read_csv(csv_path)
    assert len(frame) == small_config.synthetic.n_patients * 2
    assert set(frame.columns) >= {"image_id", "patient_id", "eye", "grade"}
    assert len(list(image_dir.glob("*.jpg"))) == len(frame)


def test_discover_images_finds_every_photograph(small_config, synthetic_dataset):
    frame = discover_images(small_config.raw_dir)
    assert len(frame) == small_config.synthetic.n_patients * 2
    assert frame["image_id"].is_unique
    assert frame["image_id"].str.islower().all()


def test_manifest_joins_labels_from_the_csv_not_the_folder(small_config, synthetic_dataset):
    manifest, meta = build_manifest(small_config)
    assert meta["n_images_without_label"] == 0
    assert meta["n_labels_without_image"] == 0
    assert manifest["grade"].nunique() >= 2
    # All images live in one directory: a folder-derived label would be constant.
    assert manifest["image_path"].map(lambda p: p.replace("\\", "/").rsplit("/", 2)[-2]).nunique() == 1


def test_manifest_derives_patient_groups(small_config, synthetic_dataset):
    manifest, meta = build_manifest(small_config)
    assert meta["patient_level_grouping"] is True
    assert meta["n_groups"] == small_config.synthetic.n_patients
    assert (manifest.groupby("group").size() == 2).all()


def test_single_class_labels_are_rejected(tmp_path, small_config, synthetic_dataset):
    """The exact scenario of labels taken from a single folder name."""
    labels = pd.read_csv(small_config.raw_dir / "labels.csv")
    labels["grade"] = 0
    broken = tmp_path / "constant_labels.csv"
    labels.to_csv(broken, index=False)

    cfg = Config.load()
    cfg.project_root = small_config.project_root
    cfg.data.dataset = "synthetic"
    cfg.paths.raw_dir = small_config.paths.raw_dir
    cfg.paths.labels_csv = str(broken)

    with pytest.raises(LabelJoinError, match="Only one class"):
        build_manifest(cfg)


def test_join_failure_message_is_actionable(tmp_path, small_config, synthetic_dataset):
    labels = pd.DataFrame({"image_id": ["nothing_matches"], "grade": [1]})
    path = tmp_path / "wrong_ids.csv"
    labels.to_csv(path, index=False)

    cfg = Config.load()
    cfg.project_root = small_config.project_root
    cfg.data.dataset = "synthetic"
    cfg.paths.raw_dir = small_config.paths.raw_dir
    cfg.paths.labels_csv = str(path)

    with pytest.raises(LabelJoinError, match="zero rows"):
        build_manifest(cfg)


def test_labels_with_file_extensions_still_join(tmp_path, small_config, synthetic_dataset):
    labels = pd.read_csv(small_config.raw_dir / "labels.csv")
    labels["image_id"] = labels["image_id"].str.upper() + ".JPG"
    path = tmp_path / "extension_ids.csv"
    labels.to_csv(path, index=False)

    cfg = Config.load()
    cfg.project_root = small_config.project_root
    cfg.data.dataset = "synthetic"
    cfg.paths.raw_dir = small_config.paths.raw_dir
    cfg.paths.labels_csv = str(path)

    manifest, meta = build_manifest(cfg)
    assert meta["n_joined"] == len(labels)


@pytest.mark.parametrize(
    ("image_id", "regex", "expected"),
    [
        ("10_left", ADAPTERS["eyepacs"].group_regex, "10"),
        ("10_right", ADAPTERS["eyepacs"].group_regex, "10"),
        ("p0007_left", ADAPTERS["synthetic"].group_regex, "p0007"),
    ],
)
def test_group_regexes_of_the_public_datasets(image_id, regex, expected):
    groups, _ = derive_groups(pd.Series([image_id]), regex)
    assert groups.iloc[0] == expected


def test_missing_group_regex_falls_back_to_one_group_per_image():
    ids = pd.Series(["a", "b", "c"])
    groups, grouped = derive_groups(ids, None)
    assert grouped is False
    assert groups.tolist() == ["a", "b", "c"]


def test_label_column_autodetection_for_an_unknown_csv(tmp_path, small_config, synthetic_dataset):
    labels = pd.read_csv(small_config.raw_dir / "labels.csv")
    labels = labels.rename(columns={"image_id": "weird_name", "grade": "severity"})
    path = tmp_path / "unknown_schema.csv"
    labels.drop(columns=["patient_id", "eye"]).to_csv(path, index=False)

    cfg = Config.load()
    cfg.project_root = small_config.project_root
    cfg.data.dataset = "generic"
    cfg.paths.raw_dir = small_config.paths.raw_dir
    cfg.paths.labels_csv = str(path)

    frame, meta = load_labels(cfg, ADAPTERS["generic"])
    assert meta["id_column"] == "weird_name"
    assert meta["label_column"] == "severity"
    assert frame["grade"].nunique() >= 2


def test_summarise_manifest_shape(small_config, manifest):
    table = summarise_manifest(manifest)
    assert {"grade", "name", "n_images", "share", "n_groups"} <= set(table.columns)
    assert np.isclose(table["share"].sum(), 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# subsampling large cohorts
# --------------------------------------------------------------------------- #
def _paired_manifest(n_patients: int = 200, seed: int = 0) -> pd.DataFrame:
    """Two eyes per patient with correlated grades - the EyePACS structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for patient in range(n_patients):
        base = int(rng.choice([0, 1, 2, 3, 4], p=[0.73, 0.07, 0.15, 0.03, 0.02]))
        for eye in ("left", "right"):
            rows.append(
                {
                    "image_id": f"{patient}_{eye}",
                    "image_path": f"/img/{patient}_{eye}.jpeg",
                    "grade": int(np.clip(base + rng.integers(-1, 2), 0, 4)),
                    "group": str(patient),
                    "source": "eyepacs",
                }
            )
    return pd.DataFrame(rows)


def test_subsample_never_splits_a_patient():
    """Sampling images instead of patients would silently break the grouping."""
    manifest = _paired_manifest()
    subset = subsample_by_patient(manifest, max_images=120, seed=1)

    assert len(subset) <= 130  # the budget is honoured up to one whole patient
    sizes = subset.groupby("group").size()
    original = manifest.groupby("group").size()
    for group, size in sizes.items():
        assert size == original[group], f"patient {group} was split"


def test_subsample_is_deterministic_and_seed_dependent():
    manifest = _paired_manifest()
    a = subsample_by_patient(manifest, 200, seed=1)
    b = subsample_by_patient(manifest, 200, seed=1)
    c = subsample_by_patient(manifest, 200, seed=2)
    assert a["image_id"].tolist() == b["image_id"].tolist()
    assert a["image_id"].tolist() != c["image_id"].tolist()


def test_subsample_keeps_the_rare_grades():
    """Round-robin over strata; a naive head() would truncate grades 3 and 4 away."""
    manifest = _paired_manifest()
    subset = subsample_by_patient(manifest, max_images=100, seed=3, strategy="balanced")
    assert subset["grade"].nunique() == manifest["grade"].nunique()
    # The severe grades must be over-represented relative to the raw prevalence.
    assert (subset["grade"] >= 3).mean() > (manifest["grade"] >= 3).mean()


def test_subsample_is_a_no_op_when_the_budget_exceeds_the_cohort():
    manifest = _paired_manifest(n_patients=10)
    subset = subsample_by_patient(manifest, max_images=10_000, seed=0)
    assert len(subset) == len(manifest)


def test_prevalence_strategy_preserves_the_class_balance():
    """The default must not silently change the thing every metric depends on."""
    manifest = _paired_manifest(n_patients=800, seed=5)
    subset = subsample_by_patient(manifest, max_images=400, seed=5, strategy="prevalence")

    full = manifest["grade"].value_counts(normalize=True).sort_index()
    part = subset["grade"].value_counts(normalize=True).sort_index().reindex(full.index).fillna(0)
    assert np.abs(full - part).max() < 0.06

    balanced = subsample_by_patient(manifest, max_images=400, seed=5, strategy="balanced")
    # The balanced draw is supposed to differ - that is its whole purpose.
    assert (balanced["grade"] >= 3).mean() > (subset["grade"] >= 3).mean()


def test_unknown_subsample_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown subsample strategy"):
        subsample_by_patient(_paired_manifest(n_patients=20), 10, seed=0, strategy="nope")


def test_manifest_signature_tracks_settings_that_change_its_contents():
    """A shared interim_dir must not let one config reuse another's manifest."""
    from dr.data.datasets import manifest_signature

    base = Config.load("configs/eyepacs.yaml")
    same = Config.load("configs/eyepacs.yaml")
    assert manifest_signature(base) == manifest_signature(same)

    for field, value in (
        ("max_images", 1234),
        ("subsample_strategy", "balanced"),
        ("group_regex", r"^(\d+)_"),
        ("dataset", "generic"),
    ):
        other = Config.load("configs/eyepacs.yaml")
        setattr(other.data, field, value)
        assert manifest_signature(other) != manifest_signature(base), field

    other = Config.load("configs/eyepacs.yaml")
    other.seed = base.seed + 1
    assert manifest_signature(other) != manifest_signature(base)


def test_build_manifest_records_its_signature(small_config, synthetic_dataset):
    from dr.data.datasets import manifest_signature

    _frame, meta = build_manifest(small_config)
    assert meta["manifest_signature"] == manifest_signature(small_config)
