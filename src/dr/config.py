"""Typed configuration objects loaded from YAML.

Typed config beats a bag of globals: every stage receives exactly the knobs it
needs, unknown keys are reported instead of silently ignored, and the resolved
configuration is serialised next to the artefacts of every run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from .utils import get_logger

LOGGER = get_logger("config")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

GRADE_NAMES: dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    labels_csv: str | None = None
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    image_glob: str = "**/*"
    id_column: str | None = None
    label_column: str | None = None
    group_regex: str | None = None
    drop_ungradable: bool = True
    max_images: int | None = None
    # "prevalence" keeps the cohort's class balance (metrics stay interpretable);
    # "balanced" enriches the rare grades at the cost of every prevalence-
    # dependent number.
    subsample_strategy: str = "prevalence"


@dataclass
class SyntheticConfig:
    n_patients: int = 160
    image_size: int = 512
    jpeg_quality: int = 92
    grade_prevalence: list[float] = field(
        default_factory=lambda: [0.49, 0.10, 0.27, 0.08, 0.06]
    )


@dataclass
class FeatureConfig:
    work_size: int = 512
    clahe_clip: float = 0.02
    clahe_kernel_fraction: float = 0.125
    vessel_sigmas: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0])
    glcm_distances: list[int] = field(default_factory=lambda: [1, 3])
    glcm_angles_deg: list[float] = field(default_factory=lambda: [0, 45, 90, 135])
    lbp_radius: int = 2
    lbp_points: int = 8
    n_zones: int = 4
    n_jobs: int = -1
    cache: bool = True


@dataclass
class QualityConfig:
    # Absolute floors: a hard safety net only. They cannot adapt to a camera,
    # so gradability is decided mainly by the cohort-relative rule below.
    min_mask_fraction: float = 0.20
    min_focus: float = 3.0e-5
    max_mean_brightness: float = 0.92
    min_mean_brightness: float = 0.06
    # Cohort-relative: flag the worst this-fraction of the cohort on brightness
    # and on focus. Set to 0 to rely on the absolute floors alone.
    drop_quantile: float = 0.01


@dataclass
class ModelingConfig:
    test_size: float = 0.25
    cv_folds: int = 5
    n_search_iter: int = 30
    nested_cv: bool = True
    nested_outer_folds: int = 5
    n_bootstrap: int = 2000
    primary_metric: str = "qwk"
    referable_threshold_grade: int = 2
    target_sensitivity: float = 0.90
    models: list[str] = field(
        default_factory=lambda: ["logreg", "random_forest", "hist_gbm", "ordinal_gbm"]
    )
    permutation_repeats: int = 20


@dataclass
class Config:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    project_root: str = str(PROJECT_ROOT)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        payload: dict[str, Any] = {}
        path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        else:
            LOGGER.warning("Config %s not found - falling back to built-in defaults", path)

        if overrides:
            payload = _deep_update(payload, overrides)
        return _from_dict(cls, payload, prefix="")

    # -- convenience ------------------------------------------------------- #
    def resolve(self, relative: str | Path) -> Path:
        """Interpret a config path.

        ``~`` expands to the home directory, absolute paths are taken as they
        are, and anything else is relative to the project root.  The ``~`` case
        is what lets a config that points at a downloaded cohort stay portable:
        hard-coding ``C:/Users/someone/dr-data`` would make the file useless to
        everyone but its author.
        """
        text = str(relative)
        if text.startswith("~"):
            return Path(text).expanduser()
        path = Path(text)
        return path if path.is_absolute() else Path(self.project_root) / path

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.paths.raw_dir)

    @property
    def interim_dir(self) -> Path:
        return self.resolve(self.paths.interim_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.paths.processed_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.paths.artifacts_dir)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.paths.reports_dir)

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def manifest_path(self) -> Path:
        return self.interim_dir / "manifest.csv"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def feature_cache_path(self) -> Path:
        return self.interim_dir / "feature_cache.jsonl"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _from_dict(cls: type, payload: dict[str, Any], prefix: str) -> Any:
    """Build a (possibly nested) dataclass, warning about unknown keys.

    ``from __future__ import annotations`` turns field types into strings, so
    the nested dataclasses are recovered with ``get_type_hints`` rather than
    ``field.type``.
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected a mapping for '{prefix or cls.__name__}', got {type(payload).__name__}"
        )

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key in sorted(set(payload) - known):
        LOGGER.warning("Ignoring unknown config key '%s%s'", prefix, key)

    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in payload:
            continue
        value = payload[name]
        hint = hints.get(name)
        if isinstance(hint, type) and is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value or {}, prefix=f"{prefix}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)
