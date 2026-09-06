"""Fundus preprocessing: field-of-view detection, cropping, illumination repair.

Fundus photographs are dominated by acquisition nuisance - the camera aperture,
uneven flash illumination, pupil size, focus.  Every downstream biomarker is
computed *after* the pipeline below, so that a feature reflects the retina and
not the camera:

1. detect the circular field of view and keep only the retina,
2. crop to the field of view and resize to a fixed working resolution,
3. correct illumination (Ben Graham's local-average subtraction),
4. build a CLAHE-equalised green channel - the channel with the highest
   vessel/lesion contrast - for structure analysis,
5. score image quality, so ungradable frames can be excluded rather than
   silently poisoning the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk
from skimage.transform import resize as sk_resize

from ..config import FeatureConfig, QualityConfig

__all__ = ["RetinaImage", "preprocess_image", "retina_mask", "load_rgb", "remove_small"]


def remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Drop connected components smaller than ``min_size`` pixels.

    ``skimage.morphology.remove_small_objects`` renamed its threshold parameter
    in 0.26 (and changed it from ``<`` to ``<=``); this shim keeps the project
    working on both the old and the new API without emitting warnings.
    """
    if not binary.any():
        return binary
    labels, n_labels = ndi.label(binary)
    if n_labels == 0:
        return binary
    counts = np.bincount(labels.ravel())
    keep = counts >= int(min_size)
    keep[0] = False
    return keep[labels]


@dataclass
class RetinaImage:
    """Everything the descriptors need, computed once per image."""

    image_id: str
    path: str
    rgb: np.ndarray            # (S, S, 3) float32 in [0, 1]
    mask: np.ndarray           # (S, S) bool, retina field of view
    green: np.ndarray          # (S, S) float32, raw green channel
    gray: np.ndarray           # (S, S) float32, luminance
    clahe: np.ndarray          # (S, S) float32, CLAHE-equalised green
    illum: np.ndarray          # (S, S, 3) float32, illumination-corrected RGB
    center: tuple[float, float]
    radius: float
    quality: dict[str, Any] = field(default_factory=dict)
    # Memo for maps that several descriptor families need (the Frangi filter in
    # particular is the single most expensive step and was previously computed
    # twice per image).
    cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def n_mask_pixels(self) -> int:
        return int(self.mask.sum())

    def inscribed_square(self) -> tuple[slice, slice]:
        """Largest axis-aligned square fully inside the field of view.

        Texture descriptors (GLCM, LBP, entropy) need a rectangular support; if
        they were run on the padded frame, the black surround would dominate.
        """
        cy, cx = self.center
        half = max(int(self.radius / np.sqrt(2.0)) - 2, 8)
        y0 = int(np.clip(cy - half, 0, self.size - 1))
        y1 = int(np.clip(cy + half, y0 + 8, self.size))
        x0 = int(np.clip(cx - half, 0, self.size - 1))
        x1 = int(np.clip(cx + half, x0 + 8, self.size))
        return slice(y0, y1), slice(x0, x1)

    def zone_masks(self, n_zones: int) -> list[np.ndarray]:
        """Concentric annuli from the fovea outwards, clipped to the retina."""
        yy, xx = np.ogrid[: self.size, : self.size]
        cy, cx = self.center
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(self.radius, 1.0)
        edges = np.linspace(0.0, 1.0, n_zones + 1)
        zones = []
        for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            inner = dist >= lo
            # The radius is an area-equivalent estimate and the field of view is
            # never a perfect circle, so the outermost annulus is left open;
            # otherwise a handful of rim pixels would belong to no zone at all.
            outer = np.ones_like(dist, dtype=bool) if index == n_zones - 1 else dist < hi
            zones.append(self.mask & inner & outer)
        return zones


# --------------------------------------------------------------------------- #
# loading and field-of-view detection
# --------------------------------------------------------------------------- #
def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image as float32 RGB in [0, 1]."""
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def retina_mask(rgb: np.ndarray, threshold: float = 0.075) -> np.ndarray:
    """Binary mask of the circular field of view.

    The surround of a fundus photograph is near-black, so a low fixed threshold
    on the per-pixel channel maximum is both simple and robust; the mask is then
    cleaned by hole filling and by keeping the largest connected component.
    """
    brightness = rgb.max(axis=2)
    mask = brightness > threshold

    if mask.mean() < 0.05:  # very dark acquisition - relax the threshold
        finite = brightness[np.isfinite(brightness)]
        mask = brightness > max(float(np.percentile(finite, 60)) * 0.5, 0.02)

    mask = ndi.binary_fill_holes(mask)
    mask = remove_small(mask, min_size=max(int(0.01 * mask.size), 64))

    labels, n_labels = ndi.label(mask)
    if n_labels > 1:
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        mask = labels == int(counts.argmax())
    elif n_labels == 0:
        mask = np.ones(rgb.shape[:2], dtype=bool)
    return mask.astype(bool)


def _crop_to_mask(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the field-of-view bounding box and pad to a square."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return rgb, mask

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    rgb, mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    height, width = mask.shape
    side = max(height, width)
    pad_y, pad_x = side - height, side - width
    pad = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2))
    rgb = np.pad(rgb, pad + ((0, 0),), mode="constant")
    mask = np.pad(mask, pad, mode="constant")
    return rgb, mask


def _resize(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = sk_resize(rgb, (size, size), order=1, anti_aliasing=True, preserve_range=True)
    mask = sk_resize(mask.astype(np.float32), (size, size), order=1, anti_aliasing=False) > 0.5
    return rgb.astype(np.float32), mask


# --------------------------------------------------------------------------- #
# enhancement
# --------------------------------------------------------------------------- #
def ben_graham(rgb: np.ndarray, mask: np.ndarray, sigma_fraction: float = 1 / 30) -> np.ndarray:
    """Subtract the local average colour (Graham, Kaggle DR 2015 winner).

    Removes the flash gradient that otherwise makes two photographs of the same
    retina look completely different.
    """
    sigma = max(rgb.shape[0] * sigma_fraction, 1.0)
    background = np.dstack(
        [ndi.gaussian_filter(rgb[..., c], sigma=sigma, mode="nearest") for c in range(3)]
    )
    corrected = 4.0 * rgb - 4.0 * background + 0.5
    return np.clip(corrected, 0.0, 1.0).astype(np.float32) * mask[..., None]


def clahe_green(green: np.ndarray, clip: float, kernel_fraction: float) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the green channel."""
    kernel = max(int(green.shape[0] * kernel_fraction), 8)
    values = np.clip(green, 0.0, 1.0)
    equalised = equalize_adapthist(values, kernel_size=kernel, clip_limit=float(clip), nbins=256)
    return equalised.astype(np.float32)


# --------------------------------------------------------------------------- #
# quality control
# --------------------------------------------------------------------------- #
def quality_metrics(
    gray: np.ndarray, mask: np.ndarray, qcfg: QualityConfig
) -> dict[str, Any]:
    """Cheap, interpretable gradability proxies.

    ``focus`` is the variance of the Laplacian inside the field of view - the
    standard blur proxy; a defocused fundus photograph has almost no
    high-frequency energy.
    """
    mask_fraction = float(mask.mean())
    inside = gray[mask] if mask.any() else gray.ravel()
    brightness = float(inside.mean()) if inside.size else 0.0
    laplacian = ndi.laplace(gray)
    focus = float(laplacian[mask].var()) if mask.any() else 0.0

    reasons: list[str] = []
    if mask_fraction < qcfg.min_mask_fraction:
        reasons.append("field_of_view_too_small")
    if focus < qcfg.min_focus:
        reasons.append("out_of_focus")
    if brightness > qcfg.max_mean_brightness:
        reasons.append("overexposed")
    if brightness < qcfg.min_mean_brightness:
        reasons.append("underexposed")

    return {
        "qc_mask_fraction": mask_fraction,
        "qc_brightness": brightness,
        "qc_focus": focus,
        "gradable": not reasons,
        "qc_reasons": ";".join(reasons),
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def preprocess_image(
    path: str | Path,
    fcfg: FeatureConfig,
    qcfg: QualityConfig,
    image_id: str | None = None,
) -> RetinaImage:
    """Run the full preprocessing chain for one photograph."""
    path = Path(path)
    rgb = load_rgb(path)
    mask = retina_mask(rgb)
    rgb, mask = _crop_to_mask(rgb, mask)
    rgb, mask = _resize(rgb, mask, int(fcfg.work_size))

    # Trim the aperture rim: its steep gradient would be picked up as "vessels".
    erosion_radius = max(int(0.012 * fcfg.work_size), 2)
    eroded = ndi.binary_erosion(mask, structure=disk(erosion_radius), border_value=0)
    if eroded.sum() > 0.5 * mask.sum():
        mask = eroded
    rgb = rgb * mask[..., None]

    green = rgb[..., 1].astype(np.float32)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

    coords = np.argwhere(mask)
    if coords.size:
        center = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        radius = float(np.sqrt(mask.sum() / np.pi))
    else:  # degenerate image - fall back to the whole frame
        center = (fcfg.work_size / 2.0, fcfg.work_size / 2.0)
        radius = fcfg.work_size / 2.0

    return RetinaImage(
        image_id=image_id or path.stem,
        path=str(path),
        rgb=rgb,
        mask=mask,
        green=green,
        gray=gray,
        clahe=clahe_green(green, fcfg.clahe_clip, fcfg.clahe_kernel_fraction),
        illum=ben_graham(rgb, mask),
        center=center,
        radius=radius,
        quality=quality_metrics(gray, mask, qcfg),
    )
