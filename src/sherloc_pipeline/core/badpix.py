"""Anomalous detector channels for the optional spectral annotation overlay.

The web layer reads absolute CCD channel indices from
``sherloc_pipeline/data/badpix_channels.csv``. The overlay marks listed
channels without changing spectral values or the cosmic-ray despiking mask.

This pipeline's dark-frame test flagged the tier 1 channels. Their source is
``dark_veto`` for the test alone or ``both`` for channels also in the list of
anomalous pixels (Jakubek et al. 2025, Supplemental Tables S3 and S4).
The paper lists the 11 tier 2 channels (``jb25``) and the dark-frame test
did not flag them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

#: Allowed ``source`` attribution values in the curated asset.
VALID_BADPIX_SOURCES = {"dark_veto", "jb25", "both", "g5_eps"}


@dataclass(frozen=True)
class BadpixChannel:
    """One anomalous detector channel.

    ``channel`` is the absolute 0-based CCD channel index (0..2147); the web
    layer maps it to a served-array position for the requested region view.
    """

    region: str
    channel: int
    tier: int
    source: str


def _parse_badpix_csv(text: str) -> Tuple[BadpixChannel, ...]:
    """Parse the curated CSV text into immutable records.

    ``#``-prefixed provenance header lines are skipped; the first
    non-comment line is the column header (``region,channel,tier,source``).
    Returns a tuple (immutable, so the module-level cache cannot be mutated by
    a caller).
    """
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    reader = csv.DictReader(data_lines)
    records: List[BadpixChannel] = []
    for row in reader:
        records.append(
            BadpixChannel(
                region=row["region"].strip(),
                channel=int(row["channel"]),
                tier=int(row["tier"]),
                source=row["source"].strip(),
            )
        )
    return tuple(records)


@lru_cache(maxsize=1)
def load_badpix_channels() -> Tuple[BadpixChannel, ...]:
    """Return the anomalous channel records (parsed once, cached).

    The asset never changes at runtime, so the parse is memoized for the
    process lifetime. Resolved via :mod:`importlib.resources` so the wheel
    install path and editable/source checkouts behave identically.
    """
    from importlib import resources

    target = resources.files("sherloc_pipeline.data") / "badpix_channels.csv"
    text = target.read_text(encoding="utf-8")
    return _parse_badpix_csv(text)
