"""A procedural fundus-photograph generator, so the pipeline is runnable offline.

Public retinopathy datasets cannot be redistributed, which usually leaves a
repository that nobody can execute.  This module renders anatomically plausible
retinal images whose *lesion burden* follows the clinical grading scale:

    0  No DR                 clean retina
    1  Mild NPDR             a few microaneurysms
    2  Moderate NPDR         microaneurysms + dot/blot haemorrhages + exudates
    3  Severe NPDR           many haemorrhages, venous beading, cotton-wool spots
    4  Proliferative DR      neovascularisation on top of severe changes

Three properties are deliberate and make the demo scientifically honest:

* **Overlapping classes.**  Lesion counts are Poisson draws whose means overlap
  between neighbouring grades, and 15 % of eyes are rendered with a neighbouring
  grade's burden (grader disagreement).  A perfect score is therefore impossible.
* **Nuisance variation.**  Illumination, colour cast, defocus, aperture size and
  rotation vary per image, so brightness alone cannot solve the task.
* **Patient structure.**  Both eyes of a patient share appearance parameters and
  correlated grades, which is what makes patient-level splitting necessary - and
  lets the pipeline *measure* the optimism of a naive random split.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..config import Config
from ..utils import ensure_dir, get_logger

LOGGER = get_logger("synthetic")

# Mean lesion counts per grade: (microaneurysms, haemorrhages, exudates,
# cotton-wool spots, neovascular fronds).  Neighbouring rows overlap on purpose.
LESION_MEANS: dict[int, tuple[float, float, float, float, float]] = {
    0: (0.3, 0.05, 0.05, 0.0, 0.0),
    1: (4.0, 0.4, 0.2, 0.0, 0.0),
    2: (11.0, 4.0, 5.0, 0.4, 0.0),
    3: (22.0, 12.0, 9.0, 2.5, 0.0),
    4: (26.0, 15.0, 10.0, 3.5, 2.2),
}


@dataclass
class EyeAppearance:
    """Per-patient camera / anatomy parameters shared by both eyes."""

    base_hue: np.ndarray          # RGB multipliers of the retinal background
    brightness: float
    vignette: float
    aperture: float               # retina radius as a fraction of the frame
    vessel_alpha: float
    vessel_seed: int
    noise: float
    blur: float


# --------------------------------------------------------------------------- #
# low level drawing primitives
# --------------------------------------------------------------------------- #
def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(size, dtype=np.float32)
    return np.meshgrid(axis, axis, indexing="ij")


def _stamp_blob(canvas: np.ndarray, cy: float, cx: float, radius: float, value: float = 1.0) -> None:
    """Add a soft-edged radial blob to a 2-D float canvas (in place)."""
    size = canvas.shape[0]
    radius = max(float(radius), 0.6)
    reach = int(math.ceil(radius * 2.5))
    y0, y1 = max(int(cy) - reach, 0), min(int(cy) + reach + 1, size)
    x0, x1 = max(int(cx) - reach, 0), min(int(cx) + reach + 1, canvas.shape[1])
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32), np.arange(x0, x1, dtype=np.float32), indexing="ij"
    )
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    patch = np.exp(-dist2 / (2.0 * (radius / 1.6) ** 2), dtype=np.float32)
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], value * patch)


def _stamp_segment(
    canvas: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], width: float
) -> None:
    """Draw a smooth line segment of the given width onto a 2-D float canvas."""
    (y0, x0), (y1, x1) = p0, p1
    length = math.hypot(y1 - y0, x1 - x0)
    steps = max(int(length), 1)
    for t in np.linspace(0.0, 1.0, steps + 1, dtype=np.float32):
        _stamp_blob(canvas, y0 + t * (y1 - y0), x0 + t * (x1 - x0), width)


def _draw_vessel_tree(
    size: int,
    disc: tuple[float, float],
    generator: np.random.Generator,
    n_main: int = 8,
    tortuosity: float = 1.0,
) -> np.ndarray:
    """Recursively grow a branching vessel tree away from the optic disc."""
    canvas = np.zeros((size, size), dtype=np.float32)
    cy, cx = disc
    centre = size / 2.0

    # Main arcades leave the disc towards the macula, roughly up and down.
    towards_centre = math.atan2(centre - cy, centre - cx)
    base_angles = towards_centre + np.linspace(-math.pi, math.pi, n_main, endpoint=False)
    base_angles = base_angles + generator.normal(0.0, 0.12, size=n_main)

    def grow(y: float, x: float, angle: float, width: float, length: float, depth: int) -> None:
        if depth > 4 or width < 0.55 or length < 6:
            return
        steps = max(int(length / 9), 2)
        step_len = length / steps
        for _ in range(steps):
            angle += generator.normal(0.0, 0.16 * tortuosity)
            ny = y + step_len * math.sin(angle)
            nx = x + step_len * math.cos(angle)
            _stamp_segment(canvas, (y, x), (ny, nx), width)
            y, x = ny, nx
            if not (0 <= y < size and 0 <= x < size):
                return
            if generator.random() < 0.16:
                grow(
                    y,
                    x,
                    angle + generator.choice([-1.0, 1.0]) * generator.uniform(0.4, 0.9),
                    width * generator.uniform(0.55, 0.75),
                    length * generator.uniform(0.35, 0.6),
                    depth + 1,
                )
        grow(y, x, angle, width * 0.72, length * 0.55, depth + 1)

    for angle in base_angles:
        grow(
            cy + 4 * math.sin(angle),
            cx + 4 * math.cos(angle),
            float(angle),
            width=generator.uniform(2.4, 3.4),
            length=size * generator.uniform(0.30, 0.42),
            depth=0,
        )
    return np.clip(canvas, 0.0, 1.0)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=(sigma, sigma, 0), mode="nearest")


# --------------------------------------------------------------------------- #
# image synthesis
# --------------------------------------------------------------------------- #
def _sample_appearance(generator: np.random.Generator) -> EyeAppearance:
    return EyeAppearance(
        base_hue=np.array(
            [
                generator.uniform(0.78, 1.00),
                generator.uniform(0.34, 0.52),
                generator.uniform(0.14, 0.28),
            ],
            dtype=np.float32,
        ),
        brightness=float(generator.uniform(0.72, 1.12)),
        vignette=float(generator.uniform(0.35, 0.75)),
        aperture=float(generator.uniform(0.42, 0.48)),
        vessel_alpha=float(generator.uniform(0.45, 0.70)),
        vessel_seed=int(generator.integers(0, 2**31 - 1)),
        noise=float(generator.uniform(0.006, 0.022)),
        blur=float(generator.uniform(0.0, 1.4)),
    )


def render_fundus(
    grade: int,
    appearance: EyeAppearance,
    laterality: str,
    size: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Render one RGB fundus image (uint8) with the lesion burden of ``grade``."""
    yy, xx = _coordinate_grid(size)
    centre = size / 2.0
    radius = appearance.aperture * size

    dist = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)
    retina = (dist <= radius).astype(np.float32)
    # Soft aperture edge, like a real camera stop.
    edge = np.clip((radius - dist) / 6.0, 0.0, 1.0).astype(np.float32)

    # Background: radial illumination falloff plus a smooth low-frequency cast.
    falloff = 1.0 - appearance.vignette * (dist / max(radius, 1.0)) ** 2
    cast_y = generator.uniform(-0.12, 0.12)
    cast_x = generator.uniform(-0.12, 0.12)
    cast = 1.0 + cast_y * (yy - centre) / size + cast_x * (xx - centre) / size
    intensity = np.clip(appearance.brightness * falloff * cast, 0.0, 1.6).astype(np.float32)

    image = intensity[..., None] * appearance.base_hue[None, None, :]

    # Optic disc: temporal side depends on laterality (right eye -> disc left).
    sign = 1.0 if laterality == "left" else -1.0
    disc_y = centre + generator.normal(0.0, 0.02 * size)
    disc_x = centre + sign * 0.30 * size + generator.normal(0.0, 0.02 * size)
    disc_r = 0.085 * size * generator.uniform(0.85, 1.15)

    disc = np.zeros((size, size), dtype=np.float32)
    _stamp_blob(disc, disc_y, disc_x, disc_r * 1.15, 1.0)
    disc = np.clip(disc, 0.0, 1.0)
    image += disc[..., None] * np.array([0.30, 0.32, 0.24], dtype=np.float32)

    # Macula: darker pigmented area on the opposite side of the disc.
    macula = np.zeros((size, size), dtype=np.float32)
    _stamp_blob(macula, centre + generator.normal(0, 0.01 * size), centre - sign * 0.05 * size,
                0.13 * size, 1.0)
    image -= macula[..., None] * np.array([0.16, 0.10, 0.05], dtype=np.float32)

    # Vessels: darkest in the green channel, which is why green is used for
    # vessel and lesion analysis in the feature extractor.
    vessel_rng = np.random.default_rng(appearance.vessel_seed + (0 if laterality == "left" else 1))
    vessels = _draw_vessel_tree(
        size,
        (disc_y, disc_x),
        vessel_rng,
        n_main=int(generator.integers(7, 10)),
        tortuosity=1.0 + 0.5 * (grade >= 3),
    )
    if grade >= 4:  # neovascular fronds: fine, tortuous, disc-centred
        fronds = _draw_vessel_tree(
            size,
            (disc_y, disc_x),
            np.random.default_rng(int(generator.integers(0, 2**31 - 1))),
            n_main=int(generator.integers(3, 6)),
            tortuosity=3.0,
        )
        vessels = np.maximum(vessels, 0.75 * fronds)

    attenuation = appearance.vessel_alpha * vessels
    image *= 1.0 - attenuation[..., None] * np.array([0.55, 0.80, 0.70], dtype=np.float32)

    # ---------------- lesions ---------------- #
    n_ma, n_he, n_ex, n_cw, n_nv = (
        int(generator.poisson(m)) for m in LESION_MEANS[int(grade)]
    )

    def random_position(min_r: float = 0.10, max_r: float = 0.93) -> tuple[float, float]:
        theta = generator.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(generator.uniform(min_r**2, max_r**2))
        return centre + r * math.sin(theta), centre + r * math.cos(theta)

    dark = np.zeros((size, size), dtype=np.float32)
    bright = np.zeros((size, size), dtype=np.float32)

    for _ in range(n_ma):  # microaneurysms: 1-2 px dark red dots
        y, x = random_position()
        _stamp_blob(dark, y, x, generator.uniform(1.2, 2.6) * size / 512, 0.6)
    for _ in range(n_he):  # dot/blot haemorrhages
        y, x = random_position()
        _stamp_blob(dark, y, x, generator.uniform(3.0, 9.0) * size / 512, 0.85)
    for _ in range(n_ex):  # hard exudates, often clustered near the macula
        y, x = random_position(0.05, 0.75)
        cluster = int(generator.integers(1, 4))
        for _ in range(cluster):
            _stamp_blob(
                bright,
                y + generator.normal(0, 4 * size / 512),
                x + generator.normal(0, 4 * size / 512),
                generator.uniform(1.5, 4.5) * size / 512,
                0.9,
            )
    for _ in range(n_cw):  # cotton-wool spots: larger, softer, paler
        y, x = random_position(0.10, 0.80)
        _stamp_blob(bright, y, x, generator.uniform(6.0, 12.0) * size / 512, 0.5)
    for _ in range(n_nv):  # pre-retinal haemorrhage next to new vessels
        y, x = random_position(0.05, 0.5)
        _stamp_blob(dark, y, x, generator.uniform(8.0, 16.0) * size / 512, 0.7)

    image *= 1.0 - dark[..., None] * np.array([0.45, 0.75, 0.72], dtype=np.float32)
    image += bright[..., None] * np.array([0.42, 0.40, 0.16], dtype=np.float32)

    # ---------------- camera effects ---------------- #
    image = np.clip(image, 0.0, 1.0)
    image = _gaussian_blur(image, appearance.blur)
    image += generator.normal(0.0, appearance.noise, size=image.shape).astype(np.float32)
    image *= (retina * edge)[..., None]
    image = np.clip(image, 0.0, 1.0)

    return (image * 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# dataset generation
# --------------------------------------------------------------------------- #
def _render_patient(
    patient_index: int,
    seed: int,
    prevalence: np.ndarray,
    size: int,
    quality: int,
    image_dir: Path,
) -> list[dict]:
    """Render both eyes of one patient. Independent per patient => embarrassingly parallel."""
    generator = np.random.default_rng([seed, patient_index])
    patient_id = f"P{patient_index:04d}"
    appearance = _sample_appearance(generator)
    patient_grade = int(generator.choice(len(prevalence), p=prevalence))

    records: list[dict] = []
    for laterality in ("left", "right"):
        # Fellow eyes correlate but are not identical.
        if generator.random() < 0.72:
            grade = patient_grade
        else:
            grade = int(np.clip(patient_grade + generator.choice([-1, 1]), 0, 4))

        # Grader disagreement: render a neighbouring grade's lesion burden while
        # keeping the recorded label. This caps the achievable ceiling below 1.0.
        render_grade = grade
        if generator.random() < 0.15:
            render_grade = int(np.clip(grade + generator.choice([-1, 1]), 0, 4))

        image = render_fundus(render_grade, appearance, laterality, size, generator)
        image_id = f"{patient_id}_{laterality}"
        Image.fromarray(image).save(image_dir / f"{image_id}.jpg", quality=quality)

        records.append(
            {"image_id": image_id, "patient_id": patient_id, "eye": laterality, "grade": grade}
        )
    return records


def generate_dataset(cfg: Config, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Render the demo cohort and write ``images/`` plus ``labels.csv``.

    Returns the image directory and the CSV path.  The CSV - not the folder
    layout - is the single source of truth for the labels, mirroring how the
    real datasets are distributed.
    """
    from joblib import Parallel, delayed

    out_dir = Path(out_dir) if out_dir is not None else cfg.raw_dir
    image_dir = ensure_dir(out_dir / "images")

    prevalence = np.asarray(cfg.synthetic.grade_prevalence, dtype=float)
    prevalence = prevalence / prevalence.sum()
    size = int(cfg.synthetic.image_size)
    n_patients = int(cfg.synthetic.n_patients)

    batches = Parallel(n_jobs=cfg.features.n_jobs, verbose=0)(
        delayed(_render_patient)(
            index, int(cfg.seed), prevalence, size, int(cfg.synthetic.jpeg_quality), image_dir
        )
        for index in range(n_patients)
    )
    records = [row for batch in batches for row in batch]

    frame = pd.DataFrame.from_records(records).sort_values("image_id").reset_index(drop=True)
    csv_path = out_dir / "labels.csv"
    ensure_dir(out_dir)
    frame.to_csv(csv_path, index=False)

    LOGGER.info(
        "Synthetic cohort: %d images / %d patients -> %s",
        len(frame),
        frame["patient_id"].nunique(),
        image_dir,
    )
    LOGGER.info("Grade distribution: %s", frame["grade"].value_counts().sort_index().to_dict())
    return image_dir, csv_path
