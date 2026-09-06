"""Turn a folder of fundus images plus a grading CSV into a clean manifest.

This module exists because of the single most common way a retinopathy project
silently fails: the labels are taken from the *folder name*.  Public retinopathy
datasets (EyePACS, APTOS-2019, Messidor-2, IDRiD) ship every image in one flat
directory and keep the severity grade in a CSV.  Deriving the label from the
parent directory therefore yields exactly one class, and any accuracy reported
afterwards is meaningless.

``build_manifest`` joins images to their CSV grades by file stem, derives a
patient/eye group key for leakage-free splitting, and refuses to return a
single-class table.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..config import GRADE_NAMES, Config
from ..utils import get_logger

LOGGER = get_logger("data")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".jpe", ".tif", ".tiff", ".bmp", ".ppm")


class LabelJoinError(RuntimeError):
    """Raised when images and labels cannot be matched into a usable dataset."""


@dataclass(frozen=True)
class DatasetAdapter:
    """Knows how a public dataset names its id / label columns and its patients."""

    name: str
    id_candidates: tuple[str, ...]
    label_candidates: tuple[str, ...]
    default_labels_csv: tuple[str, ...] = ()
    group_regex: str | None = None
    description: str = ""

    def find_column(self, columns: Sequence[str], candidates: Iterable[str]) -> str | None:
        lowered = {str(c).strip().lower(): c for c in columns}
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None


ADAPTERS: dict[str, DatasetAdapter] = {
    "aptos": DatasetAdapter(
        name="aptos",
        id_candidates=("id_code", "image", "image_id", "id"),
        label_candidates=("diagnosis", "level", "grade", "dr_grade"),
        default_labels_csv=("train.csv", "train_1.csv"),
        group_regex=None,  # one image per subject in the public release
        description="APTOS 2019 Blindness Detection (train.csv: id_code, diagnosis)",
    ),
    "eyepacs": DatasetAdapter(
        name="eyepacs",
        id_candidates=("image", "id_code", "image_id"),
        label_candidates=("level", "diagnosis", "grade"),
        default_labels_csv=("trainLabels.csv", "trainLabels_cropped.csv"),
        # "10_left" / "10_right" belong to the same patient -> group on "10"
        group_regex=r"^(\d+)_(?:left|right)$",
        description="Kaggle Diabetic Retinopathy Detection / EyePACS (image, level)",
    ),
    "messidor2": DatasetAdapter(
        name="messidor2",
        id_candidates=("image_id", "image", "id", "id_code"),
        label_candidates=(
            "adjudicated_dr_grade",
            "dr_grade",
            "retinopathy grade",
            "grade",
            "diagnosis",
        ),
        default_labels_csv=("messidor_data.csv", "messidor-2.csv"),
        group_regex=r"^(.*?)_[A-Za-z]*\d*$",
        description="Messidor-2 (image_id, adjudicated_dr_grade)",
    ),
    "idrid": DatasetAdapter(
        name="idrid",
        id_candidates=("image name", "image_name", "image", "id_code"),
        label_candidates=("retinopathy grade", "grade", "dr_grade", "diagnosis"),
        default_labels_csv=(
            "IDRiD_Disease_Grading_Training_Labels.csv",
            "a. IDRiD_Disease Grading_Training Labels.csv",
        ),
        group_regex=None,
        description="IDRiD disease grading (Image name, Retinopathy grade)",
    ),
    "generic": DatasetAdapter(
        name="generic",
        id_candidates=("image_id", "id_code", "image", "filename", "file", "name", "id"),
        label_candidates=("grade", "diagnosis", "level", "label", "dr_grade", "class"),
        default_labels_csv=("labels.csv",),
        group_regex=None,
        description="Any folder of images plus a CSV with an id column and a grade column",
    ),
    "synthetic": DatasetAdapter(
        name="synthetic",
        id_candidates=("image_id",),
        label_candidates=("grade",),
        default_labels_csv=("labels.csv",),
        group_regex=r"^(P\d+)_(?:left|right)$",
        description="Procedurally generated fundus images shipped with this repository",
    ),
}


# --------------------------------------------------------------------------- #
# image discovery
# --------------------------------------------------------------------------- #
def discover_images(root: str | Path, pattern: str = "**/*") -> pd.DataFrame:
    """Recursively collect image files under ``root``.

    Returns a frame with ``image_id`` (lower-cased file stem) and ``image_path``.
    Duplicate stems are reported and the first occurrence wins, because a stem
    collision would otherwise duplicate labels during the join.
    """
    root = Path(root)
    if not root.exists():
        raise LabelJoinError(
            f"Image directory '{root}' does not exist. "
            "Point paths.raw_dir at the folder holding the fundus photographs, "
            "or run `dr synth` to generate the bundled demo dataset."
        )

    base_pattern = pattern if pattern else "**/*"
    if base_pattern.endswith(tuple(IMAGE_EXTENSIONS)):
        base_pattern = base_pattern.rsplit(".", 1)[0] + ".*"

    files: list[Path] = [
        p
        for p in sorted(root.glob(base_pattern))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        files = [
            p
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    if not files:
        raise LabelJoinError(f"No image files ({', '.join(IMAGE_EXTENSIONS)}) found under '{root}'.")

    frame = pd.DataFrame(
        {
            "image_id": [p.stem.strip().lower() for p in files],
            "image_path": [str(p) for p in files],
        }
    )

    duplicated = frame["image_id"].duplicated(keep=False)
    if duplicated.any():
        n_dup = int(frame.loc[duplicated, "image_id"].nunique())
        LOGGER.warning(
            "%d file stems occur more than once under %s; keeping the first occurrence of each",
            n_dup,
            root,
        )
        frame = frame.drop_duplicates(subset="image_id", keep="first")

    LOGGER.info("Discovered %d images under %s", len(frame), root)
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# label loading
# --------------------------------------------------------------------------- #
def _locate_labels_csv(cfg: Config, adapter: DatasetAdapter) -> Path:
    if cfg.paths.labels_csv:
        path = cfg.resolve(cfg.paths.labels_csv)
        if not path.exists():
            raise LabelJoinError(f"labels_csv '{path}' does not exist.")
        return path

    search_roots = [cfg.raw_dir, cfg.raw_dir.parent, Path(cfg.project_root) / "data"]
    for name in adapter.default_labels_csv:
        for root in search_roots:
            candidate = root / name
            if candidate.exists():
                return candidate

    found = sorted({p for root in search_roots if root.exists() for p in root.glob("*.csv")})
    if len(found) == 1:
        LOGGER.info("Using the only CSV found next to the images: %s", found[0])
        return found[0]

    raise LabelJoinError(
        "Could not locate the grading CSV.\n"
        f"Dataset '{adapter.name}' normally ships as {adapter.default_labels_csv or '(unknown)'}.\n"
        f"Searched: {[str(r) for r in search_roots]}\n"
        "Set paths.labels_csv in the config (or --labels-csv on the command line).\n"
        "Do NOT fall back to folder names for labels: public retinopathy datasets keep "
        "all images in one directory, which would collapse the task to a single class."
    )


def load_labels(cfg: Config, adapter: DatasetAdapter | None = None) -> tuple[pd.DataFrame, dict]:
    """Read the grading CSV and normalise it to ``image_id`` / ``grade``."""
    adapter = adapter or ADAPTERS[cfg.data.dataset]
    csv_path = _locate_labels_csv(cfg, adapter)
    raw = pd.read_csv(csv_path)
    if raw.empty:
        raise LabelJoinError(f"Label file '{csv_path}' is empty.")

    id_col = cfg.data.id_column or adapter.find_column(raw.columns, adapter.id_candidates)
    label_col = cfg.data.label_column or adapter.find_column(raw.columns, adapter.label_candidates)

    if id_col is None or id_col not in raw.columns:
        id_col = _guess_id_column(raw)
    if label_col is None or label_col not in raw.columns:
        label_col = _guess_label_column(raw, exclude=id_col)

    if id_col is None or label_col is None:
        raise LabelJoinError(
            f"Could not determine id/label columns in '{csv_path}'. "
            f"Columns present: {list(raw.columns)}. "
            "Set data.id_column and data.label_column explicitly."
        )

    labels = pd.DataFrame(
        {
            "image_id": raw[id_col].astype(str).str.strip().str.lower(),
            "grade_raw": raw[label_col],
        }
    )
    # Ids in CSVs sometimes carry the file extension - strip it for the join.
    labels["image_id"] = labels["image_id"].str.replace(
        r"\.(png|jpe?g|tiff?|bmp|ppm)$", "", regex=True
    )
    labels["grade"] = pd.to_numeric(labels["grade_raw"], errors="coerce")

    n_bad = int(labels["grade"].isna().sum())
    if n_bad:
        LOGGER.warning("Dropping %d rows with a non-numeric grade", n_bad)
    labels = labels.dropna(subset=["grade"]).copy()
    labels["grade"] = labels["grade"].astype(int)

    before = len(labels)
    labels = labels.drop_duplicates(subset="image_id", keep="first")
    if len(labels) < before:
        LOGGER.warning("Dropped %d duplicated image ids in the label file", before - len(labels))

    meta = {
        "labels_csv": str(csv_path),
        "id_column": str(id_col),
        "label_column": str(label_col),
        "n_label_rows": int(len(labels)),
    }
    LOGGER.info(
        "Labels: %s (id='%s', grade='%s', %d rows)", csv_path.name, id_col, label_col, len(labels)
    )
    return labels[["image_id", "grade"]], meta


def _guess_id_column(raw: pd.DataFrame) -> str | None:
    """Fall back: the most unique text-like column is almost always the id."""
    best, best_ratio = None, 0.0
    for column in raw.columns:
        series = raw[column]
        ratio = series.nunique(dropna=True) / max(len(series), 1)
        looks_like_id = series.dtype == object or ratio > 0.9
        if looks_like_id and ratio > best_ratio:
            best, best_ratio = column, ratio
    if best is not None:
        LOGGER.warning("Guessed id column '%s' (uniqueness %.2f)", best, best_ratio)
    return best


def _guess_label_column(raw: pd.DataFrame, exclude: str | None) -> str | None:
    """Fall back: a small-cardinality integer column is the grade."""
    for column in raw.columns:
        if column == exclude:
            continue
        series = pd.to_numeric(raw[column], errors="coerce")
        if series.notna().mean() < 0.9:
            continue
        values = series.dropna()
        if values.empty:
            continue
        if (values % 1 == 0).all() and 2 <= values.nunique() <= 10 and values.min() >= 0:
            LOGGER.warning("Guessed label column '%s'", column)
            return column
    return None


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def derive_groups(image_ids: pd.Series, group_regex: str | None) -> tuple[pd.Series, bool]:
    """Map image ids to patient groups.

    Returns the group series and whether a real (multi-image) grouping was found.
    Falling back to "every image is its own group" is legitimate for APTOS but
    must be visible in the report, because it removes the protection against
    two eyes of the same patient landing on both sides of the split.
    """
    if not group_regex:
        return image_ids.copy(), False

    # Image ids are lower-cased during discovery so that the CSV join is
    # case-insensitive; the grouping regex has to be too.
    pattern = re.compile(group_regex, re.IGNORECASE)

    def _match(value: str) -> str:
        m = pattern.match(value)
        if not m:
            return value
        return m.group(1) if m.groups() else m.group(0)

    groups = image_ids.map(_match)
    grouped = bool(groups.nunique() < len(groups))
    if not grouped:
        LOGGER.warning(
            "group_regex '%s' produced one group per image; splits degrade to image level",
            group_regex,
        )
    return groups, grouped


# --------------------------------------------------------------------------- #
# subsampling
# --------------------------------------------------------------------------- #
def subsample_by_patient(
    manifest: pd.DataFrame,
    max_images: int,
    seed: int,
    strategy: str = "prevalence",
) -> pd.DataFrame:
    """Shrink a large cohort by dropping whole patients, never single eyes.

    EyePACS holds 35 000 images; a first pass on a laptop wants a few thousand.
    Sampling *images* would be wrong twice over: it splits fellow eyes, which
    quietly destroys the patient-level grouping the whole evaluation rests on,
    and taking the head of a sorted frame biases the sample towards whichever
    patient ids happen to sort first.  Patients are therefore sampled whole and
    the draw is seeded, so the subsample is reproducible.

    Two strategies, and the choice is not cosmetic:

    ``prevalence`` (default)
        A uniform random draw over patients, so the class distribution of the
        subsample matches the cohort.  Positive predictive value, negative
        predictive value, the specificity attainable at a target sensitivity and
        quadratic weighted kappa are *all* functions of prevalence, so this is
        the only setting under which those headline numbers transfer to the full
        cohort.
    ``balanced``
        Round-robin over the strata, which enriches the rare severe grades.
        Useful when the question is "can the descriptors separate grade 4 at
        all", but it inflates every prevalence-dependent metric, and the report
        has to say so.
    """
    if max_images >= len(manifest) or manifest.empty:
        return manifest.reset_index(drop=True)
    if strategy not in {"prevalence", "balanced"}:
        raise ValueError(f"Unknown subsample strategy '{strategy}'")

    generator = np.random.default_rng(seed)
    worst = manifest.groupby("group", observed=True)["grade"].max()
    sizes = manifest.groupby("group", observed=True).size()

    chosen: list = []
    total = 0
    if strategy == "prevalence":
        for group in generator.permutation(sizes.index.to_numpy()):
            chosen.append(group)
            total += int(sizes[group])
            if total >= max_images:
                break
    else:
        order: dict[int, list] = {}
        for stratum in sorted(worst.unique()):
            members = worst.index[worst == stratum].to_numpy()
            order[int(stratum)] = list(generator.permutation(members))

        while total < max_images and any(order.values()):
            for stratum in sorted(order):
                queue = order[stratum]
                if not queue:
                    continue
                group = queue.pop()
                chosen.append(group)
                total += int(sizes[group])
                if total >= max_images:
                    break

    subset = manifest[manifest["group"].isin(set(chosen))]
    kept = subset["grade"].nunique()
    if kept < manifest["grade"].nunique():
        LOGGER.warning(
            "Subsample kept only %d of %d classes; raise data.max_images",
            kept,
            manifest["grade"].nunique(),
        )
    LOGGER.info(
        "Subsampled %d images from %d patients (of %d images / %d patients), strategy=%s",
        len(subset), subset["group"].nunique(), len(manifest), manifest["group"].nunique(),
        strategy,
    )
    return subset.sort_values("image_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def manifest_signature(cfg: Config) -> str:
    """Fingerprint of every setting that changes what the manifest contains.

    The manifest is a cached artefact like any other, and it lives in a directory
    that two configs may legitimately share.  Without this, switching
    ``max_images`` from 6000 to null silently reuses the 6000-row manifest and
    the "full cohort" run quietly trains on a sixth of the data.
    """
    payload = json.dumps(
        {
            "dataset": cfg.data.dataset,
            "image_glob": cfg.data.image_glob,
            "id_column": cfg.data.id_column,
            "label_column": cfg.data.label_column,
            "group_regex": cfg.data.group_regex,
            "max_images": cfg.data.max_images,
            "subsample_strategy": cfg.data.subsample_strategy,
            "raw_dir": str(cfg.raw_dir),
            "labels_csv": str(cfg.paths.labels_csv),
            "seed": cfg.seed,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_manifest(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Join discovered images with CSV grades and derive group keys."""
    adapter = ADAPTERS.get(cfg.data.dataset)
    if adapter is None:
        raise LabelJoinError(
            f"Unknown dataset '{cfg.data.dataset}'. Available: {sorted(ADAPTERS)}"
        )

    images = discover_images(cfg.raw_dir, cfg.data.image_glob)
    labels, label_meta = load_labels(cfg, adapter)

    merged = images.merge(labels, on="image_id", how="inner")

    n_images, n_labels, n_joined = len(images), len(labels), len(merged)
    unmatched_images = n_images - n_joined
    unmatched_labels = n_labels - n_joined
    if n_joined == 0:
        raise LabelJoinError(
            "The image/label join produced zero rows.\n"
            f"Example image ids: {images['image_id'].head(3).tolist()}\n"
            f"Example label ids: {labels['image_id'].head(3).tolist()}\n"
            "The join is on the lower-cased file stem - make sure the CSV id column "
            "holds file names, not paths or indices."
        )
    if unmatched_images:
        LOGGER.warning("%d images have no label and were dropped", unmatched_images)
    if unmatched_labels:
        LOGGER.warning("%d label rows have no matching image file", unmatched_labels)

    group_regex = cfg.data.group_regex or adapter.group_regex
    merged["group"], grouped = derive_groups(merged["image_id"], group_regex)
    merged["source"] = adapter.name

    if cfg.data.max_images and cfg.data.max_images < len(merged):
        merged = subsample_by_patient(
            merged, cfg.data.max_images, cfg.seed, cfg.data.subsample_strategy
        )

    merged = merged.sort_values("image_id").reset_index(drop=True)

    counts = merged["grade"].value_counts().sort_index()
    if merged["grade"].nunique() < 2:
        raise LabelJoinError(
            "Only one class survived the join - training would be meaningless.\n"
            f"Class counts: {counts.to_dict()}\n"
            "This is the classic symptom of labels taken from a folder name. "
            "Use the dataset's grading CSV instead."
        )

    meta = {
        "manifest_signature": manifest_signature(cfg),
        "dataset": adapter.name,
        "n_images_found": n_images,
        "n_label_rows": n_labels,
        "n_joined": n_joined,
        "n_images_without_label": unmatched_images,
        "n_labels_without_image": unmatched_labels,
        "group_regex": group_regex,
        "patient_level_grouping": grouped,
        "n_groups": int(merged["group"].nunique()),
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        **label_meta,
    }
    return merged[["image_id", "image_path", "grade", "group", "source"]], meta


def summarise_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Human-readable class table, including grade names and prevalence."""
    counts = manifest["grade"].value_counts().sort_index()
    total = int(counts.sum())
    table = pd.DataFrame(
        {
            "grade": counts.index.astype(int),
            "name": [GRADE_NAMES.get(int(g), f"grade {int(g)}") for g in counts.index],
            "n_images": counts.to_numpy(),
            "share": (counts.to_numpy() / max(total, 1)).round(4),
        }
    )
    if "group" in manifest.columns:
        per_group = manifest.groupby("grade", observed=True)["group"].nunique().sort_index()
        table["n_groups"] = per_group.reindex(counts.index).to_numpy()
    return table.reset_index(drop=True)
