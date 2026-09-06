# Model card — fundus severity grader

Following the *Model Cards for Model Reporting* format (Mitchell et al., 2019).
Concrete numbers for the current run live in [`reports/REPORT.md`](reports/REPORT.md)
and [`artifacts/results.json`](artifacts/results.json); this card describes what the
model **is**, what it is **for**, and where it **must not** be used.

---

## 1. Model details

| | |
| --- | --- |
| Task | Ordinal classification of diabetic-retinopathy severity (grades 0–4) from a single colour fundus photograph |
| Secondary output | Continuous severity score and a binary *referable DR* decision (grade ≥ 2) |
| Input | One RGB fundus image; the field of view is detected and cropped automatically |
| Representation | ~145 handcrafted retinal descriptors (colour, exposure, vasculature, lesions, optic disc, texture, zonal profile) — no learned image features |
| Estimators compared | Majority baseline, prevalence-matched baseline, multinomial logistic regression, random forest, histogram gradient boosting, and an ordinal regressor with QWK-optimal cut points |
| Selection | Highest patient-grouped cross-validated quadratic weighted kappa on the training half; the hold-out set is never consulted for selection |
| Version | `dr 1.0.0`; every run stores its resolved config, library versions and git revision in `artifacts/results.json` |
| Licence | MIT |

## 2. Intended use

**Intended.** Methodological demonstration and teaching: how to build a leakage-free,
ordinal, interpretable pipeline for a medical-imaging screening task; a strong,
fully transparent baseline against which a deep model must justify its complexity;
a triage-prioritisation prototype **under supervision of a clinician**.

**Out of scope.**

* Any clinical decision about a real patient.
* Any use as a medical device. This software has no regulatory clearance and no
  clinical validation.
* Deployment on images from a camera, population or acquisition protocol that
  differs from the training cohort without prior external validation.
* Grading images that failed quality control — the pipeline flags them, and a
  flagged image must be re-taken, not scored.

## 3. Factors

Performance is expected to vary with:

* **Image quality** — defocus, small pupils, media opacity (cataract), flash
  artefacts. Gradability is decided *relative to the cohort* (the worst 1 % on
  brightness and on focus), with absolute floors kept only as a safety net: on real
  EyePACS the absolute floors, calibrated on synthetic images, fired on zero of
  6 000 frames while unreadable images were plainly being scored. Subtle degradation
  still degrades the descriptors.
* **Camera and field definition** — resolution, aperture size, colour rendering
  and field angle change the descriptors directly.
* **Fundus pigmentation** — retinal background colour varies with ethnicity and
  age; the colour descriptors are explicitly sensitive to it. This is a plausible
  source of performance disparity and has **not** been audited here for lack of
  demographic metadata.
* **Grade prevalence** — the referral operating point is derived from this
  cohort's prevalence and must be re-derived elsewhere.

## 4. Metrics

* **Quadratic weighted kappa (primary).** Penalises a 0-vs-4 error 16× more than
  a 0-vs-1 error, matching the ordinal cost structure of the grading scale.
* **Balanced accuracy and macro F1.** Class-symmetric checks that keep the rare
  severe grades visible.
* **Referable DR (grade ≥ 2): AUROC, average precision, and the specificity
  attainable at a fixed sensitivity target** (default 90 %). This is the decision
  that a screening programme actually deploys.
* **95 % bootstrap confidence intervals** on every headline number and a **paired
  bootstrap** for model-versus-baseline comparisons.

Reported both on a locked hold-out set and via nested cross-validation, so that
model-selection optimism is visible rather than hidden.

## 5. Training data

The default configuration renders a **synthetic** cohort (`dr synth`) so that the
project is runnable without redistributing patient images. Synthetic eyes carry
lesion burdens drawn from grade-dependent, deliberately overlapping distributions,
per-image nuisance variation (illumination, colour, focus, aperture), 15 % simulated
grader disagreement, and two correlated eyes per patient.

For real work, the pipeline ships adapters for **EyePACS/Kaggle DR**, **APTOS 2019**,
**Messidor-2** and **IDRiD**; labels always come from the dataset's grading CSV,
joined to images on the file stem. See the README for the exact commands.

> Numbers obtained on the synthetic cohort measure whether the *pipeline* works.
> They are **not** evidence about real retinopathy, and the README says so in the
> same place it reports them.

The pipeline has been run end to end on two real cohorts as well as the synthetic
one, and the spread is the most important thing this card can tell you:

| cohort | hold-out QWK | referable AUROC | specificity at 90 % sensitivity |
| --- | ---: | ---: | ---: |
| APTOS 2019 (3 609 images) | 0.870 | 0.972 | 0.929 |
| EyePACS, full (34 544 eyes / 17 459 patients) | 0.503 | 0.785 | 0.375 |
| synthetic demo (320 eyes) | 0.729 | 0.933 | 0.837 |

The same descriptors and the same protocol differ by a factor of 1.7 between the two
real datasets. Performance here is a property of the **cohort** - image quality,
grading protocol, prevalence - at least as much as of the model. Any figure quoted
from this project without naming its cohort is meaningless.

Only the APTOS operating point is clinically plausible. On EyePACS, 90 % sensitivity
costs 62 % of healthy patients being referred, which is not a deployable screening
tool. Note also that APTOS supplies one image per subject, so patient-level grouping
degrades to image level there; the manifest audit reports this rather than hiding it.

## 6. Evaluation data

A group-stratified hold-out (25 % of **patients**, stratified on grade) that is
split before any fitting, tuning or inspection. Cross-validation inside the
training half is likewise patient-grouped. The pipeline additionally runs a
**leakage audit**: the same model is cross-validated once ignoring patient
grouping, and the optimism of that naive split is reported alongside the correct
number.

## 7. Ethical considerations

* **False negatives are the dangerous error.** A missed proliferative case can
  cost sight; a false positive costs an ophthalmology appointment. The operating
  point is therefore chosen at a sensitivity target, not at maximum accuracy.
* **Automation bias.** A grader who sees a model's output before their own
  reading may anchor on it. Any deployment must define whether the model reads
  first, second, or only as arbitration.
* **Equity.** Colour descriptors correlate with fundus pigmentation. Without
  demographic metadata, subgroup performance cannot be checked here — an
  unaudited risk, not an absent one.
* **Data protection.** Fundus photographs are biometric health data. `.gitignore`
  excludes `data/` and the Kaggle credentials in their entirety, and downloaded
  cohorts are written to `~/dr-data`, deliberately outside this checkout — the
  project folder may sit in a cloud-synced directory, and 35 000 patient images
  must not be uploaded as a side effect of running a training script.

## 8. Caveats and recommendations

1. The descriptors are **proxies**. `lesion_microaneurysm_count_density` counts
   small dark top-hat responses that are not vessels; it is not a lesion detector
   validated against an ophthalmologist's annotations. Figure 10 of the report
   draws the detections on the images, and residual responses at wide vessel
   junctions are visible there — the figure exists so that this failure mode is
   inspectable rather than hidden behind a number.
2. A single hold-out on a single cohort establishes internal validity only.
   External validation on a different camera and population is the minimum bar
   for any claim of transportability.
3. The model has no "I don't know" output beyond the quality-control flag.
   Adding a proper rejection option (e.g. abstaining on low-confidence scores and
   routing them to a human) is the first change any real deployment needs.
4. Retraining is required whenever the camera, the population or the grading
   protocol changes.
