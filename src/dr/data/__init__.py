"""Dataset discovery, label joining and leakage-aware manifests."""

from __future__ import annotations

from .datasets import (
    ADAPTERS,
    DatasetAdapter,
    LabelJoinError,
    build_manifest,
    discover_images,
    load_labels,
    manifest_signature,
    subsample_by_patient,
    summarise_manifest,
)

__all__ = [
    "ADAPTERS",
    "DatasetAdapter",
    "LabelJoinError",
    "build_manifest",
    "discover_images",
    "load_labels",
    "manifest_signature",
    "subsample_by_patient",
    "summarise_manifest",
]
