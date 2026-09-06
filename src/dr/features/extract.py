"""Feature-extraction orchestration: parallel, cached and fully deterministic.

The extractor is the expensive stage of the pipeline, so it is:

* **parallel** - one worker per image via joblib,
* **resumable** - every finished image is appended to a JSONL cache keyed by
  ``(path, mtime, feature-config hash)``; re-running after a crash or after
  adding images only processes what is missing,
* **fail-soft** - a corrupt photograph yields a row flagged ``extraction_ok=0``
  instead of killing the run,
* **self-describing** - every feature name maps to a human-readable definition
  that is printed in the final report.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import Config, FeatureConfig, QualityConfig
from ..utils import ensure_dir, get_logger
from . import descriptors
from .preprocess import preprocess_image

LOGGER = get_logger("features")

META_COLUMNS = ("image_id", "image_path", "grade", "group", "source")
QC_COLUMNS = ("qc_mask_fraction", "qc_brightness", "qc_focus", "gradable", "qc_reasons")

FEATURE_GROUPS: dict[str, str] = {
    "colour": "Pigmentation and white balance of the retinal background (RGB/HSV moments and ratios).",
    "exposure": "Acquisition quality: dynamic range, contrast, sharpness, over/under-exposure.",
    "vessel": "Vascular tree: Frangi vesselness, calibre, density per zone, branching, fractal dimension.",
    "lesion": "Diabetic lesions: microaneurysms, haemorrhages, hard exudates, cotton-wool spots.",
    "disc": "Optic-disc geometry and contrast, plus macular darkness.",
    "macula": "Macular region intensity and contrast.",
    "texture": "GLCM / LBP / entropy statistics of the retinal surface.",
    "zone": "Centre-to-periphery intensity profile of the fundus.",
}

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "vessel_area_fraction": "Share of the retina occupied by segmented vessels.",
    "vessel_length_fraction": "Vessel skeleton length per retina pixel (vascular density).",
    "vessel_mean_width": "Mean vessel calibre = vessel area / skeleton length.",
    "vessel_branch_density": "Branch points per unit vessel length (arborisation).",
    "vessel_endpoint_density": "Skeleton endpoints per unit length (fragmentation / drop-out).",
    "vessel_fractal_dimension": "Box-counting dimension of the vessel skeleton (complexity).",
    "lesion_microaneurysm_count_density": "Punctate dark lesions per 10 000 searched pixels.",
    "lesion_haemorrhage_count_density": "Blot-sized dark lesions per 10 000 searched pixels.",
    "lesion_exudate_small_count_density": "Punctate bright lesions (hard exudates) per 10 000 px.",
    "lesion_cottonwool_count_density": "Large soft bright lesions (cotton-wool spots) per 10 000 px.",
    "lesion_total_count_density": "Total lesion count density across all four lesion types.",
    "lesion_dark_total_area": "Retina fraction covered by dark (haemorrhagic) lesions.",
    "lesion_bright_total_area": "Retina fraction covered by bright (exudative) lesions.",
    "lesion_bright_dark_ratio": "Bright-to-dark lesion area ratio (exudative vs haemorrhagic pattern).",
    "lesion_haemorrhage_mean_eccentricity": "Elongation of the dark blobs; round blobs indicate true lesions rather than vessel remnants.",
    "lesion_red_residual_p99": "99th percentile of the red-minus-green residual after illumination correction.",
    "disc_contrast": "Optic-disc brightness relative to the retinal background.",
    "disc_offset": "Distance from the detected optic disc to the centre of the field of view.",
    "macula_contrast": "Macular darkness relative to the retinal background.",
    "exposure_focus": "Variance of the Laplacian inside the retina - the blur / gradability proxy.",
    "zone_center_periphery_ratio": "Mean brightness of the central zone over the peripheral zone.",
}


# --------------------------------------------------------------------------- #
# per-image extraction
# --------------------------------------------------------------------------- #
def extract_features_for_image(
    path: str | Path,
    fcfg: FeatureConfig,
    qcfg: QualityConfig,
    image_id: str | None = None,
) -> dict[str, Any]:
    """Preprocess one photograph and evaluate every descriptor group."""
    path = Path(path)
    identifier = image_id or path.stem
    try:
        img = preprocess_image(path, fcfg, qcfg, image_id=identifier)
        row: dict[str, Any] = {"image_id": identifier, "image_path": str(path)}
        row.update(descriptors.colour_features(img))
        row.update(descriptors.exposure_features(img))
        row.update(descriptors.vessel_features(img, fcfg))
        row.update(descriptors.lesion_features(img, fcfg))
        row.update(descriptors.disc_features(img))
        row.update(descriptors.texture_features(img, fcfg))
        row.update(descriptors.zonal_features(img, fcfg))
        row.update(img.quality)
        row["extraction_ok"] = 1
        row["extraction_error"] = ""
        return row
    except Exception as exc:  # pragma: no cover - defensive, exercised by tests
        LOGGER.warning("Feature extraction failed for %s: %s", path.name, exc)
        return {
            "image_id": identifier,
            "image_path": str(path),
            "extraction_ok": 0,
            "extraction_error": f"{type(exc).__name__}: {exc}",
            "gradable": False,
            "qc_reasons": "extraction_failed",
        }


def feature_names(row: dict[str, Any]) -> list[str]:
    """Names of the numeric descriptor columns of a feature row."""
    skip = set(META_COLUMNS) | set(QC_COLUMNS) | {"extraction_ok", "extraction_error"}
    return [key for key in row if key not in skip]


def describe_feature(name: str) -> str:
    """Human-readable definition used by the report."""
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    group = name.split("_", 1)[0]
    group = "zone" if group.startswith("zone") else group
    return FEATURE_GROUPS.get(group, "Handcrafted retinal descriptor.")


def feature_group(name: str) -> str:
    """Group a feature belongs to (used for grouped importance plots)."""
    head = name.split("_", 1)[0]
    if head.startswith("zone"):
        return "zone"
    return head if head in FEATURE_GROUPS else "other"


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #
def _code_fingerprint() -> str:
    """Hash of the extractor source files.

    The cache must be invalidated when the *descriptors change*, not only when
    the configuration does - otherwise editing a descriptor and re-running
    silently reuses features computed by the previous version, which is a
    genuinely dangerous class of bug.
    """
    digest = hashlib.sha1()
    for module in ("preprocess.py", "descriptors.py", "extract.py"):
        path = Path(__file__).with_name(module)
        try:
            digest.update(path.read_bytes())
        except OSError:  # pragma: no cover - only if the package is zipped
            digest.update(module.encode("utf-8"))
    return digest.hexdigest()[:8]


def config_signature(fcfg: FeatureConfig, qcfg: QualityConfig) -> str:
    payload = json.dumps(
        {"features": asdict(fcfg), "quality": asdict(qcfg), "code": _code_fingerprint()},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _cache_key(path: str | Path, signature: str) -> str:
    path = Path(path)
    try:
        mtime = int(path.stat().st_mtime)
    except OSError:
        mtime = 0
    return f"{path.name}|{mtime}|{signature}"


def _read_cache(cache_path: Path, signature: str) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with open(cache_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("_cache_signature") == signature:
                cached[record["_cache_key"]] = record
    return cached


def _append_cache(cache_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(cache_path.parent)
    with open(cache_path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


# --------------------------------------------------------------------------- #
# dataset level extraction
# --------------------------------------------------------------------------- #
def extract_dataset(manifest: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Extract features for every row of the manifest and merge back the labels."""
    fcfg, qcfg = cfg.features, cfg.quality
    signature = config_signature(fcfg, qcfg)
    cache_path = cfg.feature_cache_path
    cached = _read_cache(cache_path, signature) if fcfg.cache else {}

    todo: list[tuple[str, str]] = []
    reused: list[dict[str, Any]] = []
    for image_id, image_path in zip(manifest["image_id"], manifest["image_path"]):
        key = _cache_key(image_path, signature)
        if key in cached:
            reused.append(cached[key])
        else:
            todo.append((str(image_id), str(image_path)))

    if reused:
        LOGGER.info("Reusing %d cached feature rows (signature %s)", len(reused), signature)

    fresh: list[dict[str, Any]] = []
    if todo:
        n_jobs = fcfg.n_jobs if fcfg.n_jobs != 0 else 1
        LOGGER.info("Extracting features for %d images (n_jobs=%s)", len(todo), n_jobs)
        fresh = Parallel(n_jobs=n_jobs, verbose=5, batch_size=8)(
            delayed(extract_features_for_image)(path, fcfg, qcfg, image_id)
            for image_id, path in todo
        )
        if fcfg.cache:
            enriched = []
            for (image_id, path), row in zip(todo, fresh):
                record = dict(row)
                record["_cache_key"] = _cache_key(path, signature)
                record["_cache_signature"] = signature
                enriched.append(record)
            _append_cache(cache_path, enriched)

    rows = reused + fresh
    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=[c for c in ("_cache_key", "_cache_signature") if c in frame])

    # The manifest is the source of truth for labels and groups.
    merged = manifest.merge(
        frame.drop(columns=["image_path"], errors="ignore"), on="image_id", how="inner"
    )
    if len(merged) != len(manifest):
        LOGGER.warning(
            "%d manifest rows produced no feature vector", len(manifest) - len(merged)
        )

    numeric = [c for c in merged.columns if c not in META_COLUMNS + QC_COLUMNS
               and c not in ("extraction_ok", "extraction_error")]
    merged[numeric] = merged[numeric].apply(pd.to_numeric, errors="coerce")

    n_failed = int((merged.get("extraction_ok", pd.Series(1, index=merged.index)) == 0).sum())
    if n_failed:
        LOGGER.warning("%d images failed feature extraction and will be dropped", n_failed)
        merged = merged[merged["extraction_ok"] == 1].reset_index(drop=True)

    if "gradable" in merged.columns:
        merged["gradable"] = merged["gradable"].astype(bool)

    LOGGER.info("Feature matrix: %d rows x %d descriptors", len(merged), len(numeric))
    return merged.sort_values("image_id").reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Numeric descriptor columns of a feature table, in a stable order."""
    skip = set(META_COLUMNS) | set(QC_COLUMNS) | {
        "extraction_ok", "extraction_error", "split", "patient_id", "eye",
    }
    columns = [
        c
        for c in frame.columns
        if c not in skip and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return sorted(columns)


def flag_cohort_outliers(frame: pd.DataFrame, drop_quantile: float = 0.01) -> pd.DataFrame:
    """Add cohort-relative gradability flags to a feature table.

    The per-image thresholds in :mod:`dr.features.preprocess` are absolute
    constants, and an absolute constant cannot be right for every camera: a value
    tuned on one cohort is either inert on a darker one or throws away half of a
    brighter one.  On real EyePACS data the shipped floors fired on **zero** of
    6 000 images while the qualitative figure plainly showed an unreadable frame
    being scored.  The absolute floors are therefore kept only as a safety net for
    catastrophic frames, and gradability is decided relative to the cohort.

    The rule is a plain quantile: the worst ``drop_quantile`` of the cohort on
    brightness and on focus is flagged.  A median/MAD outlier test was tried first
    and does not work here - fundus brightness has a genuinely wide, skewed spread,
    so ``median - k*MAD`` falls below zero and flags nothing.  Dark frames are the
    tail of a broad distribution, not statistical outliers, and a quantile is the
    honest way to cut a tail.

    The trade-off is explicit: on a cohort where every image is readable this still
    removes ``drop_quantile`` of it.  That direction is deliberate - discarding a
    little good data costs far less than scoring an unreadable one, and published
    ungradable rates for screening cohorts are a few percent anyway.
    """
    frame = frame.copy()
    if "gradable" not in frame.columns:
        frame["gradable"] = True
    if "qc_reasons" not in frame.columns:
        frame["qc_reasons"] = ""

    frame["gradable"] = frame["gradable"].astype(bool)
    frame["qc_reasons"] = frame["qc_reasons"].fillna("").astype(str)

    if not 0.0 < drop_quantile < 0.5:
        return frame

    for column, label in (("qc_brightness", "cohort_dark"), ("qc_focus", "cohort_blurred")):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < 50:  # too few images for a quantile to mean anything
            continue

        cutoff = float(values.quantile(drop_quantile))
        candidate = (values <= cutoff).fillna(False)
        if not candidate.any():
            continue

        frame.loc[candidate, "gradable"] = False
        frame.loc[candidate, "qc_reasons"] = (
            frame.loc[candidate, "qc_reasons"].str.strip(";") + ";" + label
        ).str.strip(";")
        LOGGER.info(
            "Cohort quality control flagged %d images as '%s' (%s <= %.5f)",
            int(candidate.sum()), label, column, cutoff,
        )

    return frame


def apply_quality_control(
    frame: pd.DataFrame,
    drop_ungradable: bool,
    drop_quantile: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    """Remove ungradable frames, reporting exactly what was removed and why."""
    if "gradable" not in frame.columns:
        return frame, {"dropped": 0, "reasons": {}}

    frame = flag_cohort_outliers(frame, drop_quantile=drop_quantile)

    ungradable = frame[~frame["gradable"].astype(bool)]
    reasons: dict[str, int] = {}
    for value in ungradable.get("qc_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in str(value).split(";"):
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1

    info = {
        "dropped": int(len(ungradable)),
        "reasons": reasons,
        "drop_quantile": float(drop_quantile),
    }
    if drop_ungradable and len(ungradable):
        kept = frame[frame["gradable"].astype(bool)].reset_index(drop=True)
        LOGGER.info("Quality control removed %d ungradable images %s", len(ungradable), reasons)
        if kept["grade"].nunique() < 2:
            LOGGER.warning("Quality control would collapse the label space - keeping all images")
            return frame, {**info, "dropped": 0, "note": "QC skipped to preserve classes"}
        return kept, info
    return frame, info


def os_cpu_count() -> int:  # small helper used by the CLI banner
    return os.cpu_count() or 1
