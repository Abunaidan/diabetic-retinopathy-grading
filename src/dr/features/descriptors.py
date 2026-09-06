"""Handcrafted retinal biomarkers.

Each function maps a preprocessed :class:`~dr.features.preprocess.RetinaImage`
to a flat ``{name: float}`` dictionary.  The descriptors follow the clinical
grading criteria of diabetic retinopathy rather than being a generic image
feature dump:

``colour``     pigmentation and white balance of the fundus
``exposure``   acquisition quality, contrast and sharpness
``vessels``    calibre, density, branching and fractal complexity of the tree
``lesions``    microaneurysms, haemorrhages, hard exudates, cotton-wool spots
``disc``       optic-disc geometry and macular darkness
``texture``    GLCM / LBP / entropy statistics of the retinal surface
``zones``      centre-to-periphery intensity profile

Every function is total: degenerate inputs return finite defaults instead of
NaN, so a single corrupt photograph can never break a training run.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import frangi, sobel, threshold_otsu
from skimage.measure import label as sk_label
from skimage.measure import regionprops, shannon_entropy
from skimage.morphology import black_tophat, disk, skeletonize, white_tophat

from ..config import FeatureConfig
from .preprocess import RetinaImage, remove_small

__all__ = [
    "colour_features",
    "exposure_features",
    "vessel_features",
    "lesion_features",
    "disc_features",
    "texture_features",
    "zonal_features",
    "structure_maps",
    "LesionContext",
]

_EPS = 1e-8


def _finite(value, default: float = 0.0) -> float:
    """Coerce anything to a finite float - descriptors must never emit NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size < 8:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_skew": 0.0,
                f"{prefix}_kurt": 0.0}
    return {
        f"{prefix}_mean": _finite(values.mean()),
        f"{prefix}_std": _finite(values.std()),
        f"{prefix}_skew": _finite(skew(values)),
        f"{prefix}_kurt": _finite(kurtosis(values)),
    }


# --------------------------------------------------------------------------- #
# colour
# --------------------------------------------------------------------------- #
def colour_features(img: RetinaImage) -> dict[str, float]:
    """Pigmentation and white balance inside the field of view."""
    mask = img.mask
    out: dict[str, float] = {}
    if not mask.any():
        mask = np.ones_like(img.mask)

    channels = {"r": img.rgb[..., 0], "g": img.rgb[..., 1], "b": img.rgb[..., 2]}
    means: dict[str, float] = {}
    for name, plane in channels.items():
        values = plane[mask]
        out.update(_stats(values, f"colour_{name}"))
        means[name] = _finite(values.mean())

    out["colour_ratio_rg"] = _finite(means["r"] / (means["g"] + _EPS))
    out["colour_ratio_gb"] = _finite(means["g"] / (means["b"] + _EPS))
    out["colour_ratio_rb"] = _finite(means["r"] / (means["b"] + _EPS))

    hsv = rgb2hsv(np.clip(img.rgb, 0.0, 1.0))
    for index, name in enumerate(("hue", "sat", "val")):
        values = hsv[..., index][mask]
        out[f"colour_{name}_mean"] = _finite(values.mean())
        out[f"colour_{name}_std"] = _finite(values.std())
    return out


# --------------------------------------------------------------------------- #
# exposure / acquisition quality
# --------------------------------------------------------------------------- #
def exposure_features(img: RetinaImage) -> dict[str, float]:
    """Contrast, dynamic range and sharpness - the gradability of the frame."""
    mask = img.mask if img.mask.any() else np.ones_like(img.mask)
    green = img.green[mask]
    gray = img.gray[mask]

    out: dict[str, float] = {}
    for percentile in (1, 5, 25, 50, 75, 95, 99):
        out[f"exposure_green_p{percentile}"] = _finite(np.percentile(green, percentile))

    p75, p25 = out["exposure_green_p75"], out["exposure_green_p25"]
    out["exposure_green_iqr"] = _finite(p75 - p25)
    out["exposure_rms_contrast"] = _finite(gray.std())
    out["exposure_michelson"] = _finite(
        (out["exposure_green_p99"] - out["exposure_green_p1"])
        / (out["exposure_green_p99"] + out["exposure_green_p1"] + _EPS)
    )
    out["exposure_dark_fraction"] = _finite((gray < 0.10).mean())
    out["exposure_bright_fraction"] = _finite((gray > 0.85).mean())
    out["exposure_focus"] = _finite(img.quality.get("qc_focus", 0.0))
    out["exposure_mask_fraction"] = _finite(img.quality.get("qc_mask_fraction", 0.0))
    out["exposure_edge_density"] = _finite(sobel(img.clahe)[mask].mean())
    out["exposure_entropy_green"] = _finite(shannon_entropy(img.green[img.inscribed_square()]))
    return out


# --------------------------------------------------------------------------- #
# vasculature
# --------------------------------------------------------------------------- #
def optic_disc(img: RetinaImage) -> tuple[float, float, float]:
    """Locate the optic disc: centre ``(y, x)`` and radius, in pixels (memoised).

    The disc is the brightest compact structure in a fundus photograph, so a
    heavily smoothed maximum finds it robustly.  Its location matters twice: it
    anchors the macula, and it is where the vascular trunks are widest and the
    background brightest - the single worst region for morphological lesion
    detection, which is why every lesion channel excludes it.
    """
    if "optic_disc" not in img.cache:
        smoothed = ndi.gaussian_filter(img.gray * img.mask, sigma=max(img.size * 0.02, 1.0))
        peak = np.unravel_index(int(np.argmax(smoothed * img.mask)), smoothed.shape)
        radius = max(0.09 * img.radius * 2.0, 4.0)
        img.cache["optic_disc"] = (float(peak[0]), float(peak[1]), float(radius))
    return img.cache["optic_disc"]


def _disk_mask(shape: tuple[int, int], cy: float, cx: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2


def vesselness_map(img: RetinaImage, fcfg: FeatureConfig) -> np.ndarray:
    """Multi-scale Frangi vesselness on the CLAHE green channel (memoised).

    Vessels are *dark* ridges in a fundus photograph, hence ``black_ridges``.
    """
    if "vesselness" not in img.cache:
        response = frangi(
            img.clahe, sigmas=[float(s) for s in fcfg.vessel_sigmas], black_ridges=True
        ).astype(np.float32)
        img.cache["vesselness"] = np.nan_to_num(
            response, nan=0.0, posinf=0.0, neginf=0.0
        ) * img.mask
    return img.cache["vesselness"]


def segment_vessels(img: RetinaImage, fcfg: FeatureConfig) -> np.ndarray:
    """Binary vessel tree: Otsu on the positive vesselness, de-speckled (memoised).

    Otsu adapts to each image's own contrast, which a fixed cut-off cannot do
    across cameras; a fixed percentile would be worse still, because it would
    force every retina to have the same vessel area and destroy the feature.
    """
    if "vessel_binary" not in img.cache:
        vesselness = vesselness_map(img, fcfg)
        positive = vesselness[img.mask]
        positive = positive[positive > 0]
        try:
            threshold = float(threshold_otsu(positive)) if positive.size > 32 else 0.0
        except ValueError:
            threshold = 0.0
        binary = (vesselness > max(threshold, 1e-6)) & img.mask
        img.cache["vessel_binary"] = remove_small(
            binary, min_size=max(int(0.00004 * img.mask.sum()), 12)
        )
    return img.cache["vessel_binary"]


def _fractal_dimension(binary: np.ndarray) -> float:
    """Minkowski-Bouligand box-counting dimension of a binary structure.

    Vascular complexity drops as retinopathy progresses (capillary drop-out),
    and rises again with neovascularisation, which makes it a genuinely
    informative scalar rather than a decorative one.
    """
    if binary.sum() < 16:
        return 0.0
    size = min(binary.shape)
    max_power = int(np.floor(np.log2(size)))
    sizes = 2 ** np.arange(1, max(max_power, 2))
    counts: list[float] = []
    used: list[float] = []
    for box in sizes:
        if box >= size:
            break
        reduced = binary[: (binary.shape[0] // box) * box, : (binary.shape[1] // box) * box]
        if reduced.size == 0:
            continue
        blocks = reduced.reshape(
            reduced.shape[0] // box, box, reduced.shape[1] // box, box
        ).any(axis=(1, 3))
        count = int(blocks.sum())
        if count > 0:
            counts.append(np.log(count))
            used.append(np.log(1.0 / box))
    if len(counts) < 3:
        return 0.0
    slope = np.polyfit(used, counts, 1)[0]
    return _finite(slope)


def _neighbour_count(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant")


def vessel_features(img: RetinaImage, fcfg: FeatureConfig) -> dict[str, float]:
    """Frangi vesselness followed by morphological analysis of the tree."""
    mask = img.mask
    out: dict[str, float] = {}
    if mask.sum() < 100:
        return _empty_vessel_features(fcfg)

    vesselness = vesselness_map(img, fcfg)
    inside = vesselness[mask]
    out["vessel_response_mean"] = _finite(inside.mean())
    out["vessel_response_std"] = _finite(inside.std())
    out["vessel_response_p95"] = _finite(np.percentile(inside, 95))
    out["vessel_response_p99"] = _finite(np.percentile(inside, 99))

    binary = segment_vessels(img, fcfg)
    mask_area = float(mask.sum())
    vessel_area = float(binary.sum())
    out["vessel_area_fraction"] = _finite(vessel_area / (mask_area + _EPS))

    skeleton = skeletonize(binary)
    skeleton_length = float(skeleton.sum())
    out["vessel_length_fraction"] = _finite(skeleton_length / (mask_area + _EPS))
    out["vessel_mean_width"] = _finite(vessel_area / (skeleton_length + _EPS))

    neighbours = _neighbour_count(skeleton) * skeleton
    out["vessel_branch_density"] = _finite((neighbours >= 3).sum() / (skeleton_length + _EPS))
    out["vessel_endpoint_density"] = _finite((neighbours == 1).sum() / (skeleton_length + _EPS))
    out["vessel_fractal_dimension"] = _fractal_dimension(skeleton)

    segments = sk_label(binary)
    out["vessel_n_components"] = _finite(segments.max() / (mask_area / 1e4 + _EPS))

    for index, zone in enumerate(img.zone_masks(fcfg.n_zones)):
        area = float(zone.sum())
        out[f"vessel_density_zone{index}"] = _finite((binary & zone).sum() / (area + _EPS))

    return out


def _empty_vessel_features(fcfg: FeatureConfig) -> dict[str, float]:
    keys = [
        "vessel_response_mean", "vessel_response_std", "vessel_response_p95",
        "vessel_response_p99", "vessel_area_fraction", "vessel_length_fraction",
        "vessel_mean_width", "vessel_branch_density", "vessel_endpoint_density",
        "vessel_fractal_dimension", "vessel_n_components",
    ]
    keys += [f"vessel_density_zone{i}" for i in range(fcfg.n_zones)]
    return {key: 0.0 for key in keys}


# --------------------------------------------------------------------------- #
# lesions
# --------------------------------------------------------------------------- #
def _blob_stats(
    binary: np.ndarray,
    intensity: np.ndarray,
    center: tuple[float, float],
    radius: float,
    mask_area: float,
    prefix: str,
) -> dict[str, float]:
    """Count / size / spatial statistics of a binary lesion map."""
    out = {
        f"{prefix}_count_density": 0.0,
        f"{prefix}_area_fraction": 0.0,
        f"{prefix}_mean_area": 0.0,
        f"{prefix}_max_area": 0.0,
        f"{prefix}_mean_intensity": 0.0,
        f"{prefix}_mean_eccentricity": 0.0,
        f"{prefix}_mean_radial_position": 0.0,
        f"{prefix}_radial_dispersion": 0.0,
    }
    labels = sk_label(binary)
    if labels.max() == 0:
        return out

    props = regionprops(labels, intensity_image=intensity)
    areas = np.array([p.area for p in props], dtype=float)
    positions = np.array([p.centroid for p in props], dtype=float)
    distances = (
        np.sqrt((positions[:, 0] - center[0]) ** 2 + (positions[:, 1] - center[1]) ** 2)
        / max(radius, 1.0)
    )

    # Counts are normalised per 10 000 retina pixels so that images acquired
    # with different aperture sizes stay comparable.
    out[f"{prefix}_count_density"] = _finite(len(props) / (mask_area / 1e4 + _EPS))
    out[f"{prefix}_area_fraction"] = _finite(areas.sum() / (mask_area + _EPS))
    out[f"{prefix}_mean_area"] = _finite(areas.mean())
    out[f"{prefix}_max_area"] = _finite(areas.max())
    # ``mean_intensity`` was renamed to ``intensity_mean`` in scikit-image 0.26.
    intensities = [
        p.intensity_mean if hasattr(p, "intensity_mean") else p.mean_intensity for p in props
    ]
    out[f"{prefix}_mean_intensity"] = _finite(np.mean(intensities))
    out[f"{prefix}_mean_eccentricity"] = _finite(
        np.mean([p.eccentricity if p.area >= 5 else 0.0 for p in props])
    )
    out[f"{prefix}_mean_radial_position"] = _finite(distances.mean())
    out[f"{prefix}_radial_dispersion"] = _finite(distances.std())
    return out


class LesionContext:
    """Shared intermediate maps for lesion analysis.

    Built once and reused by :func:`lesion_features` and by the qualitative
    figure in the report, so the picture a reader sees is produced by exactly
    the same code path as the numbers the model is trained on.
    """

    __slots__ = ("responses", "zones", "vessel_mask", "min_blob", "max_blob")

    def __init__(self, img: RetinaImage, fcfg: FeatureConfig):
        mask = img.mask
        size = img.size
        small_radius = max(int(0.010 * size), 2)
        # The large element must stay well below the spacing between the major
        # arcades: a bigger one turns the *retina between the vessels* into a
        # top-hat response and floods the map with vessel-shaped artefacts.
        large_radius = max(int(0.020 * size), 4)

        # CLAHE boosts sensor and JPEG noise as much as it boosts lesions. A
        # sub-pixel Gaussian removes that noise floor while leaving the smallest
        # clinically relevant lesion (a microaneurysm is 2-4 px at this
        # resolution) intact; without it, blob counts track the camera's noise
        # level instead of the disease.
        clahe = ndi.gaussian_filter(img.clahe, sigma=0.8 * size / 512.0) * mask

        self.vessel_mask = segment_vessels(img, fcfg)
        self.responses = {
            "lesion_exudate_small": (white_tophat(clahe, disk(small_radius)), small_radius),
            "lesion_cottonwool": (white_tophat(clahe, disk(large_radius)), large_radius),
            "lesion_microaneurysm": (black_tophat(clahe, disk(small_radius)), 0),
            "lesion_haemorrhage": (black_tophat(clahe, disk(large_radius)), 0),
        }

        # A dark vessel produces a *positive* white-top-hat halo on both sides,
        # as wide as the structuring element - the classic source of phantom
        # "exudates" tracing the vascular tree.  The black top-hat has no such
        # halo: it answers on the vessel itself.  The exclusion zone is therefore
        # per channel, wide for the bright ones and tight for the dark ones,
        # rather than uniformly wide (which would blind the detector to the
        # perivascular microaneurysms that matter most).
        # The optic disc is where the vascular trunks are widest and the
        # background brightest - the worst region for morphological lesion
        # detection, and excluding it is standard practice.
        disc_y, disc_x, disc_r = optic_disc(img)
        disc_region = _disk_mask(mask.shape, disc_y, disc_x, disc_r * 1.3)

        fallback = mask & ~self.vessel_mask
        by_margin: dict[int, np.ndarray] = {}
        self.zones: dict[str, np.ndarray] = {}
        for prefix, (_response, halo) in self.responses.items():
            margin = max(halo + 2, max(int(0.010 * size), 3))
            if margin not in by_margin:  # only two distinct margins in practice
                interior = ndi.binary_erosion(mask, structure=disk(margin), border_value=0)
                vessels = ndi.binary_dilation(self.vessel_mask, structure=disk(margin))
                zone = interior & ~vessels & ~disc_region
                by_margin[margin] = zone if zone.sum() > 0.05 * mask.sum() else fallback
            self.zones[prefix] = by_margin[margin]

        self.min_blob = max(int(round(6.0 * (size / 512.0) ** 2)), 3)
        # No real lesion covers half a percent of the retina; anything that big
        # is a vessel remnant or an illumination artefact.
        self.max_blob = max(int(0.005 * mask.sum()), self.min_blob * 4)

    def zone_area(self, prefix: str) -> float:
        """Area actually searched for this lesion channel.

        Densities are normalised by this, not by the whole retina, so an eye with
        a denser vascular tree is not credited with fewer lesions.
        """
        return float(max(self.zones[prefix].sum(), 1))

    def binarise(self, prefix: str, level: float = 3.5) -> np.ndarray:
        """Threshold a top-hat response, then keep only lesion-shaped blobs.

        Two steps, each fixing a specific failure mode:

        * **median + level x MAD** over the vessel-free background.  Median/MAD
          is insensitive to the lesions themselves, so the threshold does not
          drift upwards on heavily diseased eyes - exactly what a mean+k*std
          cut-off does wrong.
        * **shape and size gating.**  Retinal lesions are compact; the residues
          of the vascular tree that survive the vessel mask are large and highly
          elongated.  Eccentricity is only trusted for blobs big enough for it to
          be meaningful.
        """
        response = self.responses[prefix][0]
        zone = self.zones[prefix]
        values = response[zone]
        if values.size < 32:
            return np.zeros_like(zone)

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        cutoff = max(median + level * mad, 0.010)
        binary = remove_small((response > cutoff) & zone, min_size=self.min_blob)
        if not binary.any():
            return binary

        labelled = sk_label(binary)
        keep = np.zeros(int(labelled.max()) + 1, dtype=bool)
        for region in regionprops(labelled):
            if region.area > self.max_blob:
                continue
            if region.area >= 12 and region.eccentricity > 0.97:
                continue
            keep[region.label] = True
        return keep[labelled]

    def response_values(self, prefix: str) -> np.ndarray:
        return self.responses[prefix][0][self.zones[prefix]]


def structure_maps(img: RetinaImage, fcfg: FeatureConfig) -> dict[str, np.ndarray]:
    """Binary vessel / bright-lesion / dark-lesion maps, for visual inspection."""
    context = LesionContext(img, fcfg)
    bright = context.binarise("lesion_exudate_small") | context.binarise("lesion_cottonwool")
    dark = context.binarise("lesion_microaneurysm") | context.binarise("lesion_haemorrhage")
    return {"vessels": context.vessel_mask, "bright": bright, "dark": dark}


def lesion_features(img: RetinaImage, fcfg: FeatureConfig) -> dict[str, float]:
    """Bright (exudate) and dark (haemorrhage / microaneurysm) lesion burden.

    Bright and dark lesions are isolated with morphological top-hats on the
    CLAHE green channel: a white top-hat keeps structures brighter than their
    surroundings and smaller than the structuring element, a black top-hat does
    the same for darker structures.  Two element sizes separate the punctate
    microaneurysm scale from the blot-haemorrhage scale.
    """
    mask = img.mask
    if mask.sum() < 100:
        return _empty_lesion_features()

    context = LesionContext(img, fcfg)

    def _response_stats(prefix: str) -> dict[str, float]:
        """Threshold-free summary of a top-hat response.

        Counting blobs needs a cut-off, and every cut-off is a hyper-parameter
        that can be tuned into the labels. These percentile statistics carry the
        same lesion-burden signal without any decision boundary at all, so the
        model gets a threshold-independent view of the same evidence.
        """
        name = f"{prefix}_response"
        values = context.response_values(prefix)
        if values.size < 32:
            return {f"{name}_p99": 0.0, f"{name}_p999": 0.0, f"{name}_top1_mean": 0.0}
        p99 = float(np.percentile(values, 99))
        return {
            f"{name}_p99": _finite(p99),
            f"{name}_p999": _finite(np.percentile(values, 99.9)),
            f"{name}_top1_mean": _finite(values[values >= p99].mean()),
        }

    out: dict[str, float] = {}
    for prefix in context.responses:
        out.update(
            _blob_stats(
                context.binarise(prefix),
                img.clahe,
                img.center,
                img.radius,
                context.zone_area(prefix),
                prefix,
            )
        )
        out.update(_response_stats(prefix))

    bright_total = out["lesion_exudate_small_area_fraction"] + out["lesion_cottonwool_area_fraction"]
    dark_total = (
        out["lesion_microaneurysm_area_fraction"] + out["lesion_haemorrhage_area_fraction"]
    )
    out["lesion_bright_total_area"] = _finite(bright_total)
    out["lesion_dark_total_area"] = _finite(dark_total)
    out["lesion_bright_dark_ratio"] = _finite(bright_total / (dark_total + _EPS))
    out["lesion_total_count_density"] = _finite(
        out["lesion_exudate_small_count_density"]
        + out["lesion_cottonwool_count_density"]
        + out["lesion_microaneurysm_count_density"]
        + out["lesion_haemorrhage_count_density"]
    )

    # Red-channel residual after illumination correction: haemorrhages stay red
    # while the background is flattened away.
    red_residual = img.illum[..., 0] - img.illum[..., 1]
    out["lesion_red_residual_mean"] = _finite(red_residual[mask].mean())
    out["lesion_red_residual_p99"] = _finite(np.percentile(red_residual[mask], 99))
    return out


def _empty_lesion_features() -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in (
        "lesion_exudate_small",
        "lesion_cottonwool",
        "lesion_microaneurysm",
        "lesion_haemorrhage",
    ):
        out.update(_blob_stats(np.zeros((4, 4), dtype=bool), np.zeros((4, 4)), (0, 0), 1.0, 1.0,
                               prefix))
        out.update(
            {
                f"{prefix}_response_p99": 0.0,
                f"{prefix}_response_p999": 0.0,
                f"{prefix}_response_top1_mean": 0.0,
            }
        )
    out.update(
        {
            "lesion_bright_total_area": 0.0,
            "lesion_dark_total_area": 0.0,
            "lesion_bright_dark_ratio": 0.0,
            "lesion_total_count_density": 0.0,
            "lesion_red_residual_mean": 0.0,
            "lesion_red_residual_p99": 0.0,
        }
    )
    return out


# --------------------------------------------------------------------------- #
# optic disc and macula
# --------------------------------------------------------------------------- #
def disc_features(img: RetinaImage) -> dict[str, float]:
    """Optic-disc position/contrast and macular darkness.

    The disc is the brightest compact structure in the fundus; locating it also
    tells the extractor where the macula is (roughly 2.5 disc diameters towards
    the centre of the frame), and macular involvement drives referral decisions.
    """
    mask = img.mask
    out = {
        "disc_offset": 0.0,
        "disc_contrast": 0.0,
        "disc_area_fraction": 0.0,
        "disc_mean_intensity": 0.0,
        "macula_darkness": 0.0,
        "macula_contrast": 0.0,
    }
    if mask.sum() < 100:
        return out

    disc_y, disc_x, disc_radius = optic_disc(img)
    yy, xx = np.ogrid[: img.size, : img.size]
    disc_region = _disk_mask(mask.shape, disc_y, disc_x, disc_radius) & mask

    retina_mean = float(img.gray[mask].mean())
    disc_mean = float(img.gray[disc_region].mean()) if disc_region.any() else retina_mean

    out["disc_offset"] = _finite(
        np.hypot(disc_y - img.center[0], disc_x - img.center[1]) / max(img.radius, 1.0)
    )
    out["disc_contrast"] = _finite((disc_mean - retina_mean) / (retina_mean + _EPS))
    out["disc_mean_intensity"] = _finite(disc_mean)

    bright = (img.gray > retina_mean + 2.5 * float(img.gray[mask].std())) & mask
    out["disc_area_fraction"] = _finite(bright.sum() / (mask.sum() + _EPS))

    # Macula: opposite side of the disc, on the line towards the frame centre.
    direction = np.array([img.center[0] - disc_y, img.center[1] - disc_x], dtype=float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > _EPS else np.array([0.0, 1.0])
    macula_y = disc_y + direction[0] * 0.42 * img.radius * 2.0
    macula_x = disc_x + direction[1] * 0.42 * img.radius * 2.0
    macula_region = ((yy - macula_y) ** 2 + (xx - macula_x) ** 2) <= (0.15 * img.radius) ** 2
    macula_region &= mask
    if macula_region.any():
        macula_mean = float(img.gray[macula_region].mean())
        out["macula_darkness"] = _finite(macula_mean)
        out["macula_contrast"] = _finite((retina_mean - macula_mean) / (retina_mean + _EPS))
    return out


# --------------------------------------------------------------------------- #
# texture
# --------------------------------------------------------------------------- #
def texture_features(img: RetinaImage, fcfg: FeatureConfig) -> dict[str, float]:
    """GLCM, LBP and entropy on the largest square inside the field of view."""
    rows, cols = img.inscribed_square()
    patch = np.clip(img.clahe[rows, cols], 0.0, 1.0)
    out: dict[str, float] = {}

    levels = 32
    quantised = np.clip((patch * (levels - 1)).round().astype(np.uint8), 0, levels - 1)
    angles = [np.deg2rad(a) for a in fcfg.glcm_angles_deg]
    distances = [int(d) for d in fcfg.glcm_distances]

    if min(quantised.shape) > max(distances) + 1:
        glcm = graycomatrix(
            quantised, distances=distances, angles=angles, levels=levels,
            symmetric=True, normed=True,
        )
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
            values = graycoprops(glcm, prop)  # (n_distances, n_angles)
            for index, distance in enumerate(distances):
                # Averaging over orientations makes the descriptor rotation
                # invariant, which matters because eye alignment is arbitrary.
                out[f"texture_glcm_{prop.lower()}_d{distance}"] = _finite(values[index].mean())
    else:
        for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm"):
            for distance in distances:
                out[f"texture_glcm_{prop}_d{distance}"] = 0.0

    points, radius = int(fcfg.lbp_points), int(fcfg.lbp_radius)
    # LBP compares neighbouring pixels, so an integer image avoids ties being
    # broken by floating-point noise (scikit-image warns about exactly this).
    lbp = local_binary_pattern(
        (patch * 255.0).round().astype(np.uint8), P=points, R=radius, method="uniform"
    )
    n_bins = points + 2
    histogram, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for index, value in enumerate(histogram):
        out[f"texture_lbp_{index}"] = _finite(value)

    out["texture_entropy_clahe"] = _finite(shannon_entropy(patch))
    out["texture_entropy_illum"] = _finite(shannon_entropy(img.illum[rows, cols, 1]))
    out["texture_gradient_mean"] = _finite(sobel(patch).mean())
    out["texture_gradient_std"] = _finite(sobel(patch).std())
    return out


# --------------------------------------------------------------------------- #
# zonal profile
# --------------------------------------------------------------------------- #
def zonal_features(img: RetinaImage, fcfg: FeatureConfig) -> dict[str, float]:
    """Centre-to-periphery intensity profile, robust to aperture size."""
    out: dict[str, float] = {}
    zone_means: list[float] = []
    for index, zone in enumerate(img.zone_masks(fcfg.n_zones)):
        if zone.sum() < 16:
            out[f"zone{index}_gray_mean"] = 0.0
            out[f"zone{index}_gray_std"] = 0.0
            zone_means.append(0.0)
            continue
        values = img.gray[zone]
        out[f"zone{index}_gray_mean"] = _finite(values.mean())
        out[f"zone{index}_gray_std"] = _finite(values.std())
        zone_means.append(_finite(values.mean()))

    if zone_means:
        out["zone_center_periphery_ratio"] = _finite(
            zone_means[0] / (zone_means[-1] + _EPS)
        )
        out["zone_profile_slope"] = _finite(
            np.polyfit(np.arange(len(zone_means)), zone_means, 1)[0]
            if len(zone_means) > 1 else 0.0
        )
    return out
