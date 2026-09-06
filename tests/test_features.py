"""Preprocessing and descriptor extraction: totality, determinism, invariance."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dr.features.descriptors import structure_maps
from dr.features.extract import (
    apply_quality_control,
    describe_feature,
    extract_dataset,
    extract_features_for_image,
    feature_group,
    feature_names,
    select_feature_columns,
)
from dr.features.preprocess import preprocess_image, remove_small, retina_mask


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
def test_retina_mask_recovers_a_circular_field_of_view():
    size = 128
    yy, xx = np.ogrid[:size, :size]
    circle = ((yy - 64) ** 2 + (xx - 64) ** 2) <= 40**2
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[circle] = 0.6

    mask = retina_mask(image)
    assert mask.dtype == bool
    overlap = (mask & circle).sum() / circle.sum()
    assert overlap > 0.95
    assert mask[0, 0] == False  # noqa: E712 - the black surround must be excluded


def test_preprocess_produces_a_square_working_image(small_config, sample_image):
    img = preprocess_image(sample_image, small_config.features, small_config.quality)
    size = small_config.features.work_size
    assert img.rgb.shape == (size, size, 3)
    assert img.mask.shape == (size, size)
    assert img.clahe.shape == (size, size)
    assert 0.0 <= float(img.rgb.min()) and float(img.rgb.max()) <= 1.0
    assert 0.3 < img.mask.mean() < 0.95
    assert img.radius > size * 0.3


def test_inscribed_square_lies_inside_the_field_of_view(small_config, sample_image):
    img = preprocess_image(sample_image, small_config.features, small_config.quality)
    rows, cols = img.inscribed_square()
    assert img.mask[rows, cols].mean() > 0.98


def test_zone_masks_partition_the_retina(small_config, sample_image):
    img = preprocess_image(sample_image, small_config.features, small_config.quality)
    zones = img.zone_masks(4)
    assert len(zones) == 4
    union = np.zeros_like(img.mask)
    for zone in zones:
        assert not (union & zone).any()  # disjoint
        union |= zone
    assert union.sum() == img.mask.sum()  # exhaustive


def test_quality_control_flags_a_black_frame(small_config, tmp_path):
    path = tmp_path / "black.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(path)
    row = extract_features_for_image(path, small_config.features, small_config.quality)
    assert row["gradable"] is False or row["extraction_ok"] == 0


def test_remove_small_drops_only_tiny_components():
    binary = np.zeros((20, 20), dtype=bool)
    binary[2, 2] = True                 # area 1
    binary[10:15, 10:15] = True         # area 25
    kept = remove_small(binary, min_size=5)
    assert kept.sum() == 25
    assert not kept[2, 2]


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #
def test_every_descriptor_is_finite(small_config, sample_image):
    row = extract_features_for_image(sample_image, small_config.features, small_config.quality)
    assert row["extraction_ok"] == 1
    values = np.array([row[name] for name in feature_names(row)], dtype=float)
    assert values.size > 100
    assert np.isfinite(values).all()


def test_extraction_is_deterministic(small_config, sample_image):
    first = extract_features_for_image(sample_image, small_config.features, small_config.quality)
    second = extract_features_for_image(sample_image, small_config.features, small_config.quality)
    for name in feature_names(first):
        assert first[name] == pytest.approx(second[name], rel=1e-12, abs=1e-12), name


def test_descriptors_are_invariant_to_the_frame_padding(small_config, sample_image, tmp_path):
    """Cropping to the field of view must make border padding irrelevant."""
    original = np.asarray(Image.open(sample_image).convert("RGB"))
    padded = np.pad(original, ((40, 40), (40, 40), (0, 0)), mode="constant")
    path = tmp_path / "padded.png"
    Image.fromarray(padded).save(path)

    a = extract_features_for_image(sample_image, small_config.features, small_config.quality)
    b = extract_features_for_image(path, small_config.features, small_config.quality)

    # Global colour statistics are the cleanest invariance check: they depend on
    # the retina only, never on how much black surrounds it.
    for name in ("colour_r_mean", "colour_g_mean", "colour_b_mean", "colour_ratio_rg"):
        assert a[name] == pytest.approx(b[name], rel=0.06), name


def test_lesion_burden_increases_with_the_rendered_grade(small_config, tmp_path):
    """A sanity check on the physics of the descriptors, not on any model."""
    from dr.data.synthetic import _sample_appearance, render_fundus

    counts = {}
    for grade in (0, 4):
        values = []
        for repeat in range(3):
            generator = np.random.default_rng([grade, repeat, 5])
            image = render_fundus(
                grade, _sample_appearance(generator), "left", 256, generator
            )
            path = tmp_path / f"g{grade}_{repeat}.jpg"
            Image.fromarray(image).save(path, quality=95)
            row = extract_features_for_image(
                path, small_config.features, small_config.quality
            )
            values.append(row["lesion_total_count_density"])
        counts[grade] = float(np.mean(values))
    assert counts[4] > counts[0]


def test_structure_maps_are_boolean_and_disjoint_from_vessels(small_config, sample_image):
    img = preprocess_image(sample_image, small_config.features, small_config.quality)
    maps = structure_maps(img, small_config.features)
    assert set(maps) == {"vessels", "bright", "dark"}
    for key, value in maps.items():
        assert value.dtype == bool
        assert value.shape == img.mask.shape
    assert not (maps["bright"] & maps["vessels"]).any()
    assert not (maps["dark"] & maps["vessels"]).any()


def test_corrupt_file_is_reported_not_raised(small_config, tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    row = extract_features_for_image(path, small_config.features, small_config.quality)
    assert row["extraction_ok"] == 0
    assert row["extraction_error"]


# --------------------------------------------------------------------------- #
# dataset level
# --------------------------------------------------------------------------- #
def test_extract_dataset_keeps_labels_and_groups(small_config, manifest):
    features = extract_dataset(manifest, small_config)
    assert len(features) == len(manifest)
    assert {"image_id", "grade", "group"} <= set(features.columns)
    assert features["grade"].tolist() == manifest.sort_values("image_id")["grade"].tolist()

    columns = select_feature_columns(features)
    assert len(columns) > 100
    assert "grade" not in columns and "group" not in columns
    assert np.isfinite(features[columns].to_numpy(dtype=float)).all()


def test_quality_control_never_collapses_the_label_space(small_config, manifest):
    features = extract_dataset(manifest, small_config)
    features.loc[features.index[1:], "gradable"] = False
    kept, info = apply_quality_control(features, drop_ungradable=True)
    assert kept["grade"].nunique() >= 2
    assert "note" in info


def test_cache_signature_tracks_the_config_and_the_extractor_source(small_config):
    """A stale cache after a descriptor edit would silently corrupt a run."""
    import dataclasses

    from dr.features.extract import _code_fingerprint, config_signature

    base = config_signature(small_config.features, small_config.quality)
    other = config_signature(
        dataclasses.replace(small_config.features, work_size=small_config.features.work_size + 32),
        small_config.quality,
    )
    assert base != other
    assert config_signature(small_config.features, small_config.quality) == base

    fingerprint = _code_fingerprint()
    assert len(fingerprint) == 8 and fingerprint == _code_fingerprint()


def test_lesion_search_zones_avoid_vessels_and_the_optic_disc(small_config, sample_image):
    from dr.features.descriptors import LesionContext, optic_disc, segment_vessels

    img = preprocess_image(sample_image, small_config.features, small_config.quality)
    context = LesionContext(img, small_config.features)
    vessels = segment_vessels(img, small_config.features)
    disc_y, disc_x, _ = optic_disc(img)

    for prefix, zone in context.zones.items():
        assert not (zone & vessels).any(), prefix
        assert zone.sum() > 0.05 * img.mask.sum(), prefix
        assert not zone[int(disc_y), int(disc_x)], prefix


def test_feature_metadata_is_available_for_the_report(small_config, sample_image):
    row = extract_features_for_image(sample_image, small_config.features, small_config.quality)
    for name in feature_names(row):
        assert isinstance(describe_feature(name), str) and describe_feature(name)
        assert feature_group(name) in {
            "colour", "exposure", "vessel", "lesion", "disc", "macula", "texture", "zone", "other"
        }


# --------------------------------------------------------------------------- #
# cohort-relative quality control
# --------------------------------------------------------------------------- #
def _qc_frame(n: int = 400, seed: int = 0):
    """A cohort with a realistic broad brightness spread plus a few dark frames."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    brightness = np.clip(rng.normal(0.38, 0.11, n), 0.02, 0.95)
    focus = np.clip(rng.normal(0.0023, 0.0011, n), 1e-5, None)
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(n)],
            "grade": rng.integers(0, 5, n),
            "group": [f"g{i//2}" for i in range(n)],
            "qc_brightness": brightness,
            "qc_focus": focus,
            "gradable": True,
            "qc_reasons": "",
        }
    )


def test_cohort_quality_control_flags_the_requested_tail():
    """The absolute floors are inert on real cohorts; the quantile rule is not."""
    from dr.features.extract import flag_cohort_outliers

    frame = _qc_frame()
    flagged = flag_cohort_outliers(frame, drop_quantile=0.02)
    bad = flagged[~flagged["gradable"]]

    # Two independent 2 % tails, overlapping in part: between 2 % and 4 % total.
    assert 0.02 <= len(bad) / len(frame) <= 0.045
    assert bad["qc_brightness"].min() == frame["qc_brightness"].min()
    assert set(";".join(bad["qc_reasons"]).split(";")) <= {"cohort_dark", "cohort_blurred"}
    # Everything flagged as dark really is darker than everything kept as bright.
    dark = bad[bad["qc_reasons"].str.contains("cohort_dark")]
    assert dark["qc_brightness"].max() <= flagged[flagged["gradable"]]["qc_brightness"].min()


def test_cohort_quality_control_is_a_no_op_when_disabled():
    from dr.features.extract import flag_cohort_outliers

    frame = _qc_frame()
    assert flag_cohort_outliers(frame, drop_quantile=0.0)["gradable"].all()


def test_cohort_quality_control_needs_enough_images_to_be_meaningful():
    """A quantile over 20 images is noise, so the rule declines to fire."""
    from dr.features.extract import flag_cohort_outliers

    assert flag_cohort_outliers(_qc_frame(n=20), drop_quantile=0.05)["gradable"].all()


def test_quality_control_preserves_the_class_balance(small_config, manifest):
    """QC must not quietly become a class filter."""
    from dr.features.extract import apply_quality_control, extract_dataset

    features = extract_dataset(manifest, small_config)
    kept, info = apply_quality_control(features, drop_ungradable=True, drop_quantile=0.01)
    assert kept["grade"].nunique() == features["grade"].nunique()
    assert "drop_quantile" in info
