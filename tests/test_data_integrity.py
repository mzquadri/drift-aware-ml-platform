"""The cached file is the one every published number is computed from.

`ensure_raw` records a digest on first download rather than hard-coding one, which is a
deliberate choice: a number copied into source and never exercised is not a check. These
tests hold the other half of that bargain - that the recorded digest covers the file that
is actually read, and that it is verified every time rather than once.

Nothing here reaches the network. The cache is seeded by hand, which is also the state a
second run sees.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from mlplatform.config import CONFIG, DataConfig
from mlplatform.data import DataIntegrityError, ensure_raw

ROWS = (
    "instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,"
    "temp,atemp,hum,windspeed,casual,registered,cnt\n"
    "1,2011-01-01,1,0,1,0,0,6,0,1,0.24,0.2879,0.81,0,3,13,16\n"
    "2,2011-01-01,1,0,1,1,0,6,0,1,0.22,0.2727,0.80,0,8,32,40\n"
)


def seeded(tmp_path: Path, body: str = ROWS) -> DataConfig:
    """A cache directory in the state a second run finds: csv plus lock."""
    cfg = dataclasses.replace(CONFIG.data, cache_dir=tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Bytes, not text: write_text translates newlines on Windows, so the digest of the
    # string and the digest of what lands on disk would differ for reasons that have
    # nothing to do with what is being tested.
    payload = body.encode("utf-8")
    csv = tmp_path / cfg.member
    csv.write_bytes(payload)
    (tmp_path / "archive.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="utf-8"
    )
    return cfg


def test_an_untouched_cache_loads_without_reaching_the_network(tmp_path: Path) -> None:
    cfg = seeded(tmp_path)

    assert ensure_raw(cfg).read_bytes() == ROWS.encode("utf-8")


def test_an_edited_cache_file_is_refused(tmp_path: Path) -> None:
    # The failure this exists for. A row appended to hour.csv changes every figure the
    # README publishes, and the digest was recorded against the archive rather than
    # against this file, so nothing noticed.
    cfg = seeded(tmp_path)
    csv = tmp_path / cfg.member
    extra = "3,2011-01-01,1,0,1,2,0,6,0,1,0.2,0.2,0.8,0,0,0,99999\n"
    csv.write_bytes((ROWS + extra).encode("utf-8"))

    with pytest.raises(DataIntegrityError, match=r"hour\.csv"):
        ensure_raw(cfg)


def test_a_truncated_cache_file_is_refused(tmp_path: Path) -> None:
    cfg = seeded(tmp_path)
    (tmp_path / cfg.member).write_bytes((ROWS.rsplit("\n", 2)[0] + "\n").encode("utf-8"))

    with pytest.raises(DataIntegrityError, match=r"hour\.csv"):
        ensure_raw(cfg)


def test_the_recorded_digest_covers_the_file_that_is_read(tmp_path: Path) -> None:
    # Recording the archive's digest while reading the extracted member means the two
    # can disagree and nothing finds out.
    cfg = seeded(tmp_path)
    recorded = (tmp_path / "archive.sha256").read_text(encoding="utf-8").strip()

    on_disk = hashlib.sha256((tmp_path / cfg.member).read_bytes()).hexdigest()
    assert recorded == on_disk


def test_a_missing_lock_is_refused_rather_than_silently_rewritten(tmp_path: Path) -> None:
    # Deleting the lock must not be a way to launder an edited file back in.
    cfg = seeded(tmp_path)
    (tmp_path / "archive.sha256").unlink()
    (tmp_path / cfg.member).write_bytes((ROWS + "3,x,1,0,1,2,0,6,0,1,0,0,0,0,0,0,9\n").encode())

    with pytest.raises(DataIntegrityError):
        ensure_raw(cfg)
