"""Command line interface.

    dr synth       render the bundled synthetic cohort (no download required)
    dr manifest    join images with the grading CSV and audit the labels
    dr features    extract the handcrafted descriptors (parallel, cached)
    dr train       tune, cross-validate and evaluate every model
    dr report      render the figures and REPORT.md
    dr predict     grade new photographs with the saved model
    dr all         the whole pipeline end to end

Every command shares the same configuration file and the same command-line
overrides, so a run can be reproduced from the printed banner alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .config import Config, DEFAULT_CONFIG_PATH
from .utils import (
    configure_console,
    ensure_dir,
    environment_fingerprint,
    get_logger,
    load_table,
    save_json,
    save_table,
    set_seed,
    setup_logging,
    timer,
)

LOGGER = get_logger("cli")


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr",
        description="Interpretable diabetic-retinopathy grading from fundus photographs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"dr {__version__}")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML configuration")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="aptos | eyepacs | messidor2 | idrid | generic | synthetic",
    )
    parser.add_argument("--raw-dir", default=None, help="folder containing the images")
    parser.add_argument("--labels-csv", default=None, help="CSV with the reference grades")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--group-regex", default=None, help="regex mapping image id -> patient id")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--n-patients", type=int, default=None, help="size of the synthetic cohort"
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--work-size", type=int, default=None)
    parser.add_argument("--models", default=None, help="comma separated model names")
    parser.add_argument("--search-iter", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--no-nested-cv", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("synth", help="render the synthetic demo cohort")
    subparsers.add_parser("manifest", help="join images and labels, audit the class balance")
    subparsers.add_parser("features", help="extract handcrafted descriptors")
    subparsers.add_parser("train", help="train, tune and evaluate")
    subparsers.add_parser("report", help="render figures and REPORT.md")
    subparsers.add_parser("all", help="run the entire pipeline")

    predict = subparsers.add_parser("predict", help="grade new photographs")
    predict.add_argument("images", nargs="+", help="image files or a directory")
    predict.add_argument("--out", default=None, help="write the predictions to this CSV")

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: dict[str, Any] = {
        "paths": {}, "data": {}, "features": {}, "modeling": {}, "synthetic": {}
    }
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.dataset:
        overrides["data"]["dataset"] = args.dataset
    if args.raw_dir:
        overrides["paths"]["raw_dir"] = args.raw_dir
    if args.labels_csv:
        overrides["paths"]["labels_csv"] = args.labels_csv
    if args.id_column:
        overrides["data"]["id_column"] = args.id_column
    if args.label_column:
        overrides["data"]["label_column"] = args.label_column
    if args.group_regex:
        overrides["data"]["group_regex"] = args.group_regex
    if args.max_images:
        overrides["data"]["max_images"] = args.max_images
    if args.n_patients:
        overrides["synthetic"]["n_patients"] = args.n_patients
    if args.n_jobs is not None:
        overrides["features"]["n_jobs"] = args.n_jobs
    if args.work_size:
        overrides["features"]["work_size"] = args.work_size
    if args.no_cache:
        overrides["features"]["cache"] = False
    if args.models:
        overrides["modeling"]["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.search_iter:
        overrides["modeling"]["n_search_iter"] = args.search_iter
    if args.cv_folds:
        overrides["modeling"]["cv_folds"] = args.cv_folds
    if args.bootstrap is not None:
        overrides["modeling"]["n_bootstrap"] = args.bootstrap
    if args.no_nested_cv:
        overrides["modeling"]["nested_cv"] = False

    overrides = {k: v for k, v in overrides.items() if v not in ({}, None)}
    return Config.load(args.config, overrides=overrides)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def command_synth(cfg: Config) -> None:
    from .data.synthetic import generate_dataset

    with timer("Rendering the synthetic cohort", LOGGER):
        image_dir, csv_path = generate_dataset(cfg)
    LOGGER.info("Images: %s", image_dir)
    LOGGER.info("Labels: %s", csv_path)


def command_manifest(cfg: Config) -> pd.DataFrame:
    from .data.datasets import build_manifest, summarise_manifest

    manifest, meta = build_manifest(cfg)
    ensure_dir(cfg.interim_dir)
    manifest.to_csv(cfg.manifest_path, index=False)
    save_json(meta, cfg.interim_dir / "manifest_meta.json")

    LOGGER.info("Manifest -> %s", cfg.manifest_path)
    print("\nLabel join audit")
    print("-" * 68)
    for key in (
        "dataset", "labels_csv", "id_column", "label_column", "n_images_found",
        "n_label_rows", "n_joined", "n_images_without_label", "n_labels_without_image",
        "patient_level_grouping", "n_groups",
    ):
        print(f"  {key:<26} {meta.get(key)}")
    print("\nClass distribution")
    print("-" * 68)
    print(summarise_manifest(manifest).to_string(index=False))
    print()
    return manifest


def _manifest_is_stale(cfg: Config) -> bool:
    """Should the manifest be rebuilt before extracting features?

    Two configs may share an ``interim_dir`` on purpose - it is what lets the
    full-cohort run reuse the subsampled run's feature cache.  The manifest,
    though, is config-dependent: reusing one built with ``max_images: 6000``
    for a run with ``max_images: null`` silently trains the "full cohort" on a
    sixth of the data.  So the manifest carries the fingerprint of the settings
    that produced it, and any mismatch forces a rebuild.
    """
    from .data.datasets import manifest_signature

    if not cfg.manifest_path.exists():
        LOGGER.info("No manifest found - building it first")
        return True

    meta_path = cfg.interim_dir / "manifest_meta.json"
    if not meta_path.exists():
        LOGGER.info("Manifest has no provenance record - rebuilding it")
        return True

    from .utils import load_json

    try:
        recorded = load_json(meta_path).get("manifest_signature")
    except Exception:
        recorded = None

    expected = manifest_signature(cfg)
    if recorded != expected:
        LOGGER.info(
            "Manifest was built with different data settings (%s != %s) - rebuilding it",
            recorded, expected,
        )
        return True
    return False


def command_features(cfg: Config) -> pd.DataFrame:
    from .features.extract import apply_quality_control, extract_dataset

    if _manifest_is_stale(cfg):
        command_manifest(cfg)
    manifest = pd.read_csv(cfg.manifest_path)

    with timer(f"Extracting descriptors for {len(manifest)} images", LOGGER):
        features = extract_dataset(manifest, cfg)

    features, qc_info = apply_quality_control(
        features, cfg.data.drop_ungradable, cfg.quality.drop_quantile
    )
    save_table(features, cfg.features_path)
    save_json(qc_info, cfg.processed_dir / "quality_control.json")

    LOGGER.info("Features -> %s", cfg.features_path)
    print(f"\n{len(features)} usable images, {qc_info['dropped']} rejected by quality control")
    if qc_info.get("reasons"):
        print(f"  reasons: {qc_info['reasons']}")
    print(f"  class distribution: {features['grade'].value_counts().sort_index().to_dict()}\n")
    return features


def _load_features(cfg: Config) -> pd.DataFrame:
    if not cfg.features_path.exists() and not cfg.features_path.with_suffix(".csv").exists():
        LOGGER.info("No feature table found - extracting it first")
        return command_features(cfg)
    return load_table(cfg.features_path)


def command_train(cfg: Config) -> dict:
    from .modeling.train import run_training

    features = _load_features(cfg)
    with timer("Training and evaluation", LOGGER):
        results = run_training(cfg, features)

    print_summary(results)
    return results


def command_report(cfg: Config) -> Path:
    from .modeling.report import build_report
    from .utils import load_json

    results_path = cfg.artifacts_dir / "results.json"
    if not results_path.exists():
        LOGGER.info("No results found - training first")
        results = command_train(cfg)
    else:
        results = load_json(results_path)

    features = _load_features(cfg)
    with timer("Building the report", LOGGER):
        path = build_report(cfg, features, results)
    print(f"\nReport written to {path}\n")
    return path


def command_predict(cfg: Config, images: Sequence[str], out: str | None) -> pd.DataFrame:
    from joblib import load

    from .features.extract import extract_features_for_image
    from .data.datasets import IMAGE_EXTENSIONS

    bundle_path = cfg.artifacts_dir / "model.joblib"
    if not bundle_path.exists():
        raise SystemExit(f"No trained model at {bundle_path}. Run `dr train` first.")
    bundle = load(bundle_path)

    paths: list[Path] = []
    for item in images:
        path = Path(item)
        if path.is_dir():
            paths.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.exists():
            paths.append(path)
        else:
            LOGGER.warning("Skipping missing path %s", path)
    if not paths:
        raise SystemExit("No images to score.")

    rows = [extract_features_for_image(p, cfg.features, cfg.quality) for p in paths]
    frame = pd.DataFrame(rows)
    columns = bundle["feature_names"]
    missing = [c for c in columns if c not in frame.columns]
    for column in missing:
        frame[column] = np.nan
    if missing:
        LOGGER.warning("%d descriptors missing from the new images", len(missing))

    X = frame[columns].to_numpy(dtype=float)
    model = bundle["model"]
    predictions = model.predict(X)

    from .modeling.metrics import ordinal_scores

    scores = ordinal_scores(model, X)
    threshold = bundle.get("referable_threshold")

    output = pd.DataFrame(
        {
            "image_id": frame["image_id"],
            "predicted_grade": np.asarray(predictions, dtype=int),
            "severity_score": np.round(scores, 4),
            "gradable": frame.get("gradable", pd.Series(True, index=frame.index)),
            "qc_reasons": frame.get("qc_reasons", pd.Series("", index=frame.index)),
        }
    )
    if threshold is not None and np.isfinite(threshold):
        output["refer"] = (scores >= float(threshold)).astype(int)

    destination = Path(out) if out else cfg.artifacts_dir / "predictions.csv"
    ensure_dir(destination.parent)
    output.to_csv(destination, index=False)
    print(output.to_string(index=False))
    print(f"\nPredictions -> {destination}\n")
    return output


def command_all(cfg: Config) -> None:
    from .data.datasets import IMAGE_EXTENSIONS

    has_images = cfg.raw_dir.exists() and any(
        p.suffix.lower() in IMAGE_EXTENSIONS for p in cfg.raw_dir.rglob("*") if p.is_file()
    )
    if cfg.data.dataset == "synthetic" and not has_images:
        command_synth(cfg)
    command_manifest(cfg)
    command_features(cfg)
    command_train(cfg)
    command_report(cfg)


# --------------------------------------------------------------------------- #
# reporting helpers
# --------------------------------------------------------------------------- #
def print_summary(results: dict) -> None:
    best = results["best_model"]
    rows = []
    for name, payload in results["models"].items():
        point, low, high = payload["confidence_intervals"]["qwk"]
        metrics = payload["metrics"]
        rows.append(
            {
                "model": name + (" *" if name == best else ""),
                "cv_qwk": round(payload["cv_qwk_mean"], 3),
                "test_qwk": round(point, 3),
                "qwk_95ci": f"[{low:.3f}, {high:.3f}]",
                "bal_acc": round(metrics["balanced_accuracy"], 3),
                "f1_macro": round(metrics["f1_macro"], 3),
                "accuracy": round(metrics["accuracy"], 3),
                "refer_auroc": round(metrics.get("referable_auroc", float("nan")), 3),
            }
        )
    table = pd.DataFrame(rows).sort_values("test_qwk", ascending=False)

    print("\n" + "=" * 96)
    print("RESULTS  (* = selected on cross-validated QWK, before the hold-out set was touched)")
    print("=" * 96)
    print(table.to_string(index=False))

    comparison = results["comparison_vs_baseline"]
    print(
        f"\nBest vs majority baseline: dQWK = {comparison['difference']:+.3f} "
        f"[{comparison['ci_low']:+.3f}, {comparison['ci_high']:+.3f}], "
        f"p = {comparison['p_value']:.4f}"
    )
    audit = results.get("leakage_audit")
    if audit:
        print(
            f"Splitting audit: random-split CV would report "
            f"{audit['random_cv_qwk_mean']:.3f} vs {audit['grouped_cv_qwk_mean']:.3f} "
            f"patient-grouped (optimism {audit['optimism']:+.3f})"
        )
    nested = results.get("nested_cv")
    if nested and best in nested["models"]:
        payload = nested["models"][best]
        print(
            f"Nested CV for '{best}': QWK {payload['mean']:.3f} ± {payload['std']:.3f} "
            f"over {nested['n_outer_folds']} outer folds"
        )
    importance = results.get("permutation_importance")
    if importance:
        top = ", ".join(row["feature"] for row in importance["per_feature"][:5])
        print(f"Most informative descriptors: {top}")
    print(f"Total runtime: {results.get('runtime_seconds', 0):.0f} s")
    print("=" * 96 + "\n")


def print_banner(cfg: Config, command: str) -> None:
    env = environment_fingerprint()
    print("=" * 96)
    print(f"dr {__version__} | command: {command} | dataset: {cfg.data.dataset} | seed: {cfg.seed}")
    print(
        f"python {env['python']} | numpy {env['numpy']} | scikit-learn {env['scikit_learn']}"
        + (f" | git {env['git_revision']}" if env.get("git_revision") else "")
    )
    print("=" * 96)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    setup_logging(args.log_level, logfile=ensure_dir(cfg.artifacts_dir) / "run.log")
    set_seed(cfg.seed)
    print_banner(cfg, args.command)

    try:
        if args.command == "synth":
            command_synth(cfg)
        elif args.command == "manifest":
            command_manifest(cfg)
        elif args.command == "features":
            command_features(cfg)
        elif args.command == "train":
            command_train(cfg)
        elif args.command == "report":
            command_report(cfg)
        elif args.command == "predict":
            command_predict(cfg, args.images, args.out)
        elif args.command == "all":
            command_all(cfg)
        else:  # pragma: no cover - argparse guarantees a valid command
            parser.error(f"unknown command {args.command}")
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.warning("Interrupted by the user")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
