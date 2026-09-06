"""Download a public retinopathy dataset and wire it into the pipeline.

Usage
-----
    python scripts/fetch_dataset.py eyepacs     # 35 126 images, real patient pairs
    python scripts/fetch_dataset.py aptos       # 3 662 images, one eye per subject

Both live on Kaggle and need API credentials: either the current bearer token in
``~/.kaggle/access_token`` (Settings -> API Tokens -> Generate New Token) or the
older ``~/.kaggle/kaggle.json``.  ``aptos`` is a *competition*, so its rules must
also be accepted once in the browser; ``eyepacs`` points at a public mirror of the
2015 competition images and needs only the token.

The script is idempotent: an already-downloaded, already-extracted dataset is
detected and left alone, so re-running it is free.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Deliberately outside the project tree: this checkout lives in a OneDrive-synced
# folder, and 35 000 fundus photographs must not be uploaded to anybody's cloud
# storage as a side effect of running a training script.
DEFAULT_DATA_ROOT = Path.home() / "dr-data"


@dataclass(frozen=True)
class Source:
    name: str
    kind: str                 # "dataset" or "competition"
    slug: str
    approx_gb: float
    labels_csv: str           # relative to the extraction root
    image_subdir: str         # relative to the extraction root
    config: str               # the config file to use afterwards
    note: str


SOURCES: dict[str, Source] = {
    "eyepacs": Source(
        name="eyepacs",
        kind="dataset",
        slug="tanlikesmath/diabetic-retinopathy-resized",
        approx_gb=7.3,   # measured on 2026-09-05
        labels_csv="trainLabels.csv",
        image_subdir="resized_train_cropped/resized_train_cropped",
        config="configs/eyepacs.yaml",
        note=(
            "EyePACS / Kaggle DR 2015, resized to 1024 px by a third party. "
            "35 126 images, file names '10_left' / '10_right' -> real patient pairs, "
            "which is what makes the leakage audit meaningful."
        ),
    ),
    "aptos": Source(
        name="aptos",
        kind="competition",
        slug="aptos2019-blindness-detection",
        approx_gb=10.0,
        labels_csv="train.csv",
        image_subdir="train_images",
        config="configs/aptos.yaml",
        note=(
            "APTOS 2019 Blindness Detection. 3 662 images, one eye per subject, so "
            "patient grouping degrades to image level (the pipeline says so in its "
            "manifest audit)."
        ),
    ),
}


def fail(message: str) -> None:
    print(f"\n[!] {message}\n", file=sys.stderr)
    raise SystemExit(1)


def adopt_downloaded_token(legacy: Path) -> bool:
    """Move a freshly downloaded kaggle.json into place, so the user need not.

    The legacy flow drops the file in the browser's download folder, and copying
    it by hand is the step everybody forgets.  The newest match wins, because
    Chrome names repeats ``kaggle(1).json``.
    """
    candidates: list[Path] = []
    for folder in (Path.home() / "Downloads", Path.home() / "Desktop", Path.cwd()):
        if folder.is_dir():
            candidates.extend(folder.glob("kaggle*.json"))
    if not candidates:
        return False

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not {"username", "key"} <= set(payload):
        return False

    legacy.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(newest), legacy)
    try:  # a credential: readable by its owner only
        os.chmod(legacy, 0o600)
    except OSError:
        pass
    print(f"[+] Adopted the API token from {newest.parent} -> {legacy}")
    return True


def check_credentials() -> None:
    """Find Kaggle credentials in any of the four places they can live.

    Kaggle now issues a single bearer token (``KGAT_...``) stored in
    ``~/.kaggle/access_token``; older accounts still have the username/key pair
    in ``kaggle.json``.  Both are accepted, as are the environment variables, so
    the script does not send someone hunting for a file they do not have.
    """
    home = Path.home() / ".kaggle"
    legacy = home / "kaggle.json"
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")

    if (home / "access_token").is_file() or legacy.is_file():
        return
    if config_dir and (
        (Path(config_dir) / "access_token").is_file()
        or (Path(config_dir) / "kaggle.json").is_file()
    ):
        return
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    if adopt_downloaded_token(legacy):
        return
    fail(
        "No Kaggle credentials found.\n"
        "    1. Open https://www.kaggle.com/settings  (log in first)\n"
        "    2. API Tokens -> 'Generate New Token'\n"
        f"    3. Save the token shown there to  {home / 'access_token'}\n"
        "       (an older kaggle.json left in Downloads is picked up automatically)"
    )


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3


def download(source: Source, destination: Path) -> Path:
    """Fetch the archive with the Kaggle API, unless it is already extracted."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    destination.mkdir(parents=True, exist_ok=True)
    if (destination / source.labels_csv).exists() or list(destination.rglob("trainLabels.csv")):
        print(f"[=] Already downloaded into {destination}")
        return destination

    needed = source.approx_gb * 2.2  # archive plus extracted copy
    available = free_gb(destination)
    if available < needed:
        fail(
            f"Not enough free space: {available:.0f} GB available, "
            f"~{needed:.0f} GB needed for '{source.name}' (download + extraction)."
        )

    api = KaggleApi()
    api.authenticate()

    print(f"[>] Downloading {source.kind} '{source.slug}' (~{source.approx_gb:.0f} GB)")
    print("    This takes a while and prints its own progress bar.\n")
    if source.kind == "dataset":
        api.dataset_download_files(source.slug, path=str(destination), unzip=True, quiet=False)
    else:
        try:
            api.competition_download_files(source.slug, path=str(destination), quiet=False)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part
            # Distinguish "you may not have this data" from "the network broke".
            # Blaming permissions for a dropped TLS connection sends the reader
            # off to re-accept rules they already accepted.
            text = f"{type(exc).__name__}: {exc}"
            if "403" in text or "Forbidden" in text or "Unauthorized" in text:
                fail(
                    f"Kaggle refused the download ({text}).\n"
                    "    Competition data needs two one-time steps on your own account:\n"
                    "    1. phone-verify it at https://www.kaggle.com/settings\n"
                    "    2. join the competition and accept its rules - open\n"
                    f"       https://www.kaggle.com/competitions/{source.slug}/data\n"
                    "       and click 'Join the competition'. The rules page's\n"
                    "       'Late Submission' button opens the same dialog."
                )
            fail(
                f"The download did not finish: {text}\n"
                "    This looks like a transport error, not a permissions problem.\n"
                "    Kaggle downloads resume: just run this script again and it will\n"
                "    continue from the partial file rather than start over."
            )
        archive = destination / f"{source.slug}.zip"
        if archive.exists():
            print(f"[>] Extracting {archive.name}")
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(destination)
            archive.unlink()
    return destination


def verify(source: Source, root: Path) -> tuple[Path, Path]:
    """Locate the labels CSV and the image folder, wherever the archive put them."""
    csv_path = root / source.labels_csv
    if not csv_path.exists():
        matches = sorted(root.rglob(Path(source.labels_csv).name))
        if not matches:
            fail(f"'{source.labels_csv}' not found under {root} after extraction.")
        csv_path = matches[0]

    image_dir = root / source.image_subdir
    if not image_dir.is_dir():
        candidates = []
        for folder in root.rglob("*"):
            if not folder.is_dir():
                continue
            count = sum(
                1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpeg", ".jpg", ".png")
            )
            if count > 100:
                candidates.append((count, folder))
        if not candidates:
            fail(f"No folder with images found under {root}.")
        image_dir = max(candidates)[1]

    n_images = sum(1 for p in image_dir.iterdir() if p.is_file())
    print(f"\n[=] Labels : {csv_path}")
    print(f"[=] Images : {image_dir}  ({n_images} files)")
    return csv_path, image_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=sorted(SOURCES), help="which cohort to fetch")
    parser.add_argument(
        "--dest", default=None, help="download directory (default data/external/<name>)"
    )
    args = parser.parse_args()

    source = SOURCES[args.dataset]
    destination = (
        Path(args.dest) if args.dest else DEFAULT_DATA_ROOT / source.name
    )

    print("=" * 78)
    print(f"{source.name}: {source.note}")
    print("=" * 78)

    check_credentials()
    root = download(source, destination)
    csv_path, image_dir = verify(source, root)

    print("\nReady. Run the pipeline with:\n")
    print(
        f'    python -m dr --config {source.config} '
        f'--raw-dir "{image_dir.as_posix()}" '
        f'--labels-csv "{csv_path.as_posix()}" all\n'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
