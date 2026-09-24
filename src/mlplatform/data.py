"""Fetching the dataset, checking it has not changed, and cutting it into periods.

The split is by calendar year, not at random. Training on 2011 and serving 2012
means the serving window is genuinely different from the training window, which
is what the drift monitor is supposed to notice.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .config import CONFIG, DataConfig

log = logging.getLogger(__name__)


class DataIntegrityError(RuntimeError):
    """Raised when the archive on disk is not the archive we recorded."""


@dataclass(frozen=True)
class Periods:
    """Reference is what the model learned from; current is what it now sees."""

    reference: pd.DataFrame
    current: pd.DataFrame


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(cfg: DataConfig) -> bytes:
    log.info("downloading %s", cfg.url)
    request = Request(cfg.url, headers={"User-Agent": "drift-aware-ml-platform"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def ensure_raw(cfg: DataConfig | None = None) -> Path:
    """Return the path to hour.csv, downloading it once and pinning its hash.

    The hash is recorded on first download rather than hard-coded, so the check
    is real instead of a number copied from somewhere and never verified.
    """
    cfg = cfg or CONFIG.data
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.cache_dir / cfg.member
    lock_path = cfg.cache_dir / "archive.sha256"

    if csv_path.exists() and lock_path.exists():
        return csv_path

    archive = _download(cfg)
    digest = _sha256(archive)

    if lock_path.exists():
        recorded = lock_path.read_text(encoding="utf-8").strip()
        if recorded != digest:
            raise DataIntegrityError(
                f"upstream archive changed: recorded {recorded}, got {digest}. "
                "Delete data/raw to accept the new file deliberately."
            )

    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = [n for n in zf.namelist() if n.endswith(cfg.member)]
        if not names:
            raise DataIntegrityError(f"{cfg.member} not present in the archive")
        csv_path.write_bytes(zf.read(names[0]))

    lock_path.write_text(digest + "\n", encoding="utf-8")
    log.info("wrote %s (archive sha256 %s)", csv_path, digest[:12])
    return csv_path


def load_raw(cfg: DataConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CONFIG.data
    frame = pd.read_csv(ensure_raw(cfg))
    missing = [c for c in (cfg.target, "yr") if c not in frame.columns]
    if missing:
        raise DataIntegrityError(f"expected columns absent from the file: {missing}")
    return frame


def drop_leakage(frame: pd.DataFrame, cfg: DataConfig | None = None) -> pd.DataFrame:
    """Remove the columns that would hand the model its own answer.

    casual + registered sum to cnt exactly. A model given either one scores
    almost perfectly in validation and is worthless in production. Dropping
    them here, once, is why no later stage has to remember to.
    """
    cfg = cfg or CONFIG.data
    return frame.drop(columns=[c for c in cfg.leaky_columns if c in frame.columns])


def split_periods(frame: pd.DataFrame, cfg: DataConfig | None = None) -> Periods:
    cfg = cfg or CONFIG.data
    reference = frame[frame["yr"] == cfg.reference_year].reset_index(drop=True)
    current = frame[frame["yr"] == cfg.current_year].reset_index(drop=True)
    if reference.empty or current.empty:
        raise DataIntegrityError("one of the periods is empty; check the yr column encoding")
    return Periods(reference=reference, current=current)


def train_valid_split(
    frame: pd.DataFrame, cfg: DataConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the tail of the period, not a random sample.

    Demand is a time series. A random split lets the model validate against
    hours it effectively saw neighbours of, which flatters it. Taking the last
    slice in chronological order is the honest version.
    """
    cfg = cfg or CONFIG.data
    # The file is already in chronological order, one row per hour, so the row
    # order is the time order. Keeping it means the validation slice is strictly
    # later than the training slice.
    cut = int(len(frame) * (1.0 - cfg.valid_fraction))
    if cut < 1 or cut >= len(frame):
        raise DataIntegrityError("valid_fraction leaves an empty split")
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def period_frames(cfg: DataConfig | None = None) -> Periods:
    """The one call the training and monitoring jobs both use."""
    cfg = cfg or CONFIG.data
    return split_periods(drop_leakage(load_raw(cfg), cfg), cfg)
