"""Known-noisy detector-channel annotation table (issue #9).

The pipeline ships a curated CSV of detector CCD channels flagged as
known-noisy (``sherloc_pipeline/data/badpix_channels.csv``). This module
parses that asset ONCE at import via :mod:`importlib.resources` (so editable
installs, wheel installs, and source checkouts all resolve it) and exposes the
records to the web layer.

The badpix table is an ANNOTATION asset only. It is fully separate from
cosmic-ray (CR) despiking: nothing here masks, replaces, or alters any
spectral value; it only lets an analyst surface "known-noisy channel" on
demand.

Note (2026-06-17): tier 1 is a DARK-PLANE-confirmed defect class. The earlier
epsilon=(active-dark) rate criterion was inverted — it flagged active-only REAL
Raman bands (it wrongly listed the sulfate :math:`\\nu_1` channels 129/130/131
and the carbonate :math:`\\nu_1` apex 137 near 1086.7 cm-1) and missed the true
RTS defects flanking them. Those false positives were removed; a real defect
must fire with the laser OFF (a Raman band cannot).

Tier semantics (carried through verbatim from the asset header):

- tier 1 = dark-plane-confirmed flickering / RTS defect (CR-confusable).
- tier 2 = stable hot pixels that cancel in dark subtraction (epsilon-quiet),
  documented in the published mission bad-pixel table.

Source attribution: ``g5_eps`` (epsilon-observable characterization),
``jb25`` (published mission bad-pixel table), or ``both``.
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
    """One curated known-noisy detector channel.

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
    """Return the curated known-noisy channel records (parsed once, cached).

    The asset never changes at runtime, so the parse is memoized for the
    process lifetime. Resolved via :mod:`importlib.resources` so the wheel
    install path and editable/source checkouts behave identically.
    """
    from importlib import resources

    target = resources.files("sherloc_pipeline.data") / "badpix_channels.csv"
    text = target.read_text(encoding="utf-8")
    return _parse_badpix_csv(text)
