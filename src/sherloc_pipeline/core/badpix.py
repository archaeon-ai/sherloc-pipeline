"""Display annotations for anomalous pixels (Jakubek et al. 2025, Supplemental Tables S3 and S4).

The overlay marks absolute CCD channels without changing spectral values.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class BadpixChannel:
    """Absolute 0-based CCD index and the supplemental table that lists it."""

    channel: int
    table: str


def _parse_badpix_csv(text: str) -> tuple[BadpixChannel, ...]:
    """Read channel and table pairs, rejecting malformed or repeated entries."""
    rows = csv.reader(line for line in text.splitlines() if line and not line.startswith("#"))
    if next(rows, None) != ["channel", "table"]:
        raise ValueError("Expected channel,table header")

    records = []
    seen = set()
    for value, table in rows:
        channel = int(value)
        if not 0 <= channel <= 2147:
            raise ValueError(f"CCD channel out of range: {channel}")
        if table not in {"S3", "S4"}:
            raise ValueError(f"Invalid supplemental table for channel {channel}: {table!r}")
        if channel in seen:
            raise ValueError(f"Duplicate CCD channel: {channel}")
        seen.add(channel)
        records.append(BadpixChannel(channel, table))
    return tuple(records)


@lru_cache(maxsize=1)
def load_badpix_channels() -> tuple[BadpixChannel, ...]:
    """Return the anomalous channel records (parsed once, cached).

    The asset never changes at runtime, so the parse is memoized for the
    process lifetime. Resolved via :mod:`importlib.resources` so the wheel
    install path and editable/source checkouts behave identically.
    """
    from importlib import resources

    target = resources.files("sherloc_pipeline.data") / "badpix_channels.csv"
    text = target.read_text(encoding="utf-8")
    return _parse_badpix_csv(text)
