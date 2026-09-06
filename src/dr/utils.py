"""Small cross-cutting helpers: logging, seeding, timing, JSON and environment.

Nothing here is domain specific; keeping it in one place means the scientific
modules stay readable.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

LOGGER_NAME = "dr"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def configure_console() -> None:
    """Make stdout/stderr survive non-ASCII paths on legacy Windows code pages.

    Project directories frequently contain non-Latin characters; without this a
    single ``logger.info(path)`` raises ``UnicodeEncodeError`` on a cp1252
    console and takes the whole run down.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - depends on the host terminal
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def setup_logging(level: str = "INFO", logfile: str | Path | None = None) -> logging.Logger:
    """Configure the package logger once; safe to call repeatedly."""
    configure_console()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(stream)

    if logfile is not None:
        logfile = Path(logfile)
        ensure_dir(logfile.parent)
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == logfile.resolve()
            for h in logger.handlers
        )
        if not already:
            file_handler = logging.FileHandler(logfile, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the package logger."""
    base = logging.getLogger(LOGGER_NAME)
    if not base.handlers:
        setup_logging()
    return base if name is None else base.getChild(name)


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every source of randomness this project touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int) -> np.random.Generator:
    """Preferred local generator; avoids relying on global numpy state."""
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=_json_default)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_table(df, path: str | Path) -> Path:
    """Write a DataFrame as parquet when pyarrow is available, else CSV."""
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:  # pragma: no cover - depends on optional pyarrow
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def load_table(path: str | Path):
    import pandas as pd

    path = Path(path)
    if not path.exists() and path.suffix == ".parquet":
        path = path.with_suffix(".csv")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# timing / provenance
# --------------------------------------------------------------------------- #
@contextmanager
def timer(message: str, logger: logging.Logger | None = None) -> Iterator[None]:
    log = logger or get_logger("timer")
    start = time.perf_counter()
    log.info("%s ...", message)
    try:
        yield
    finally:
        log.info("%s finished in %.1f s", message, time.perf_counter() - start)


def git_revision() -> str | None:
    """Short git hash if the project happens to live in a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def environment_fingerprint() -> dict[str, Any]:
    """Everything a reviewer needs to reproduce a number."""
    import sklearn

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "git_revision": git_revision(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for module_name in ("pandas", "scipy", "skimage"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "?")
        except Exception:
            info[module_name] = "missing"
    return info


def human_count(n: int) -> str:
    return f"{n:,}".replace(",", " ")
