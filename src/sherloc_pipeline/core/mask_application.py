"""Channel-mask application to the pipeline's spectral representations.

Cosmic-ray masks are sorted lists of absolute channel indices (0–2147) on a
region's 2148-channel plane (spec §3.3). This module maps those masks onto
the three frames the despike stage carries — the R1 normalized frame, the
fluorescence full-plane sum, and the masked-assignment R123 stitch — and
replaces flagged rows via the shared legacy interpolation
(:func:`~sherloc_pipeline.core.preprocessing.apply_mask_replacement`).

Two normative rules govern combined representations (spec §3.3):

- **Contributor-set principle** — at each channel, apply the masks of
  exactly the regions that contribute to that channel under that
  representation's construction. The contributor map is *derived* from the
  construction's own definition (config wavelength bounds for masked
  assignment), never hardcoded separately.
- **Certified-window applicability** — masks exist only inside each
  region's certified detection window ``[lo, hi)``. A contribution from a
  region outside its window is uncovered and retains legacy never-screened
  behavior; this module enforces the windows defensively even if a caller
  supplies out-of-window indices.

This module never imports onnxruntime or ``ml_despike`` — the certified
region windows are passed in by the caller, keeping mask application usable
for any mask source.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sherloc_pipeline.core.calibration import calculate_loupe_wavelength_wavenumber
from sherloc_pipeline.core.preprocessing import apply_mask_replacement

#: Masked-assignment write order of ``create_r123_spectrum``
#: (``core/data_ingestion.py``): sequential assignment within config
#: wavelength bounds, last writer wins — so R3 > R2 > R1 on overlap.
REGION_ASSIGNMENT_ORDER = ("R1", "R2", "R3")

#: Type of a per-scan mask set: ``{region: {point_index: sorted abs channel
#: indices}}``. Absent regions/points mean "no flags".
MaskSet = Mapping[str, Mapping[int, np.ndarray]]


def derive_region_channel_masks(
    config, n_channels: int = 2148
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Per-region boolean channel masks from the config wavelength bounds.

    Uses the identical arithmetic as ``create_r123_spectrum`` /
    ``restructure_raman_data`` (Loupe polynomial wavelength axis, inclusive
    ``[min, max]`` bounds), so the derived map cannot drift from the
    construction it describes.

    Args:
        config: Pipeline config object exposing ``spectral_regions``.
        n_channels: Plane width (2148 for SHERLOC CCD readouts).

    Returns:
        ``(wavelength, {"R1"|"R2"|"R3": bool ndarray of shape (n_channels,)})``
    """
    wavelength, _ = calculate_loupe_wavelength_wavenumber(n_channels)
    regions = config.spectral_regions
    bounds = {
        "R1": (regions.r1_wavelength_min, regions.r1_wavelength_max),
        "R2": (regions.r2_wavelength_min, regions.r2_wavelength_max),
        "R3": (regions.r3_wavelength_min, regions.r3_wavelength_max),
    }
    channel_masks = {
        region: (wavelength >= lo) & (wavelength <= hi)
        for region, (lo, hi) in bounds.items()
    }
    return wavelength, channel_masks


def derive_winning_region_map(
    region_channel_masks: Mapping[str, np.ndarray], n_channels: int = 2148
) -> np.ndarray:
    """Per-channel winning region for the masked-assignment construction.

    Replays the assignment order of ``create_r123_spectrum`` (R1 → R2 → R3,
    last writer wins within bounds). Channels assigned by no region get
    ``""`` (they hold zeros in the stitched frame and are never despiked).

    Returns:
        ``ndarray`` of dtype ``<U2``, shape ``(n_channels,)`` with values in
        ``{"", "R1", "R2", "R3"}``.
    """
    winner = np.full(n_channels, "", dtype="<U2")
    for region in REGION_ASSIGNMENT_ORDER:
        winner[np.asarray(region_channel_masks[region], dtype=bool)] = region
    return winner


def _point_columns(frame: pd.DataFrame) -> list:
    return [col for col in frame.columns if isinstance(col, (int, np.integer))]


def _in_window_channels(
    channels: Optional[np.ndarray], window: Tuple[int, int]
) -> np.ndarray:
    """Restrict mask channels to the region's certified window ``[lo, hi)``.

    Detector output is in-window by construction; this is the defensive
    enforcement of the certified-window restriction for masks from any other source.
    """
    if channels is None or len(channels) == 0:
        return np.empty(0, dtype=np.int64)
    arr = np.asarray(channels, dtype=np.int64)
    lo, hi = window
    return arr[(arr >= lo) & (arr < hi)]


def apply_masks_to_r1_frame(
    r1_df: pd.DataFrame,
    masks: MaskSet,
    r1_channel_mask: np.ndarray,
    region_windows: Mapping[str, Tuple[int, int]],
    interpolation_method: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply R1 masks to the R1 normalized frame.

    Frame rows are the channels selected by the R1 wavelength-bounds mask
    (``r1_mask`` in ``restructure_raman_data``), in channel order. Flagged
    channels outside the exported window are persisted upstream but not
    applicable to this frame — they are silently skipped (spec §3.3).

    Args:
        r1_df: Frame with ``raman_shift`` + integer point columns.
        masks: Per-scan mask set; only ``masks["R1"]`` is consulted.
        r1_channel_mask: Boolean (2148,) channel-selection mask used to
            build the frame (from :func:`derive_region_channel_masks`).
        region_windows: Certified detection windows ``{region: (lo, hi)}``.
        interpolation_method: Replacement interpolation method.

    Returns:
        ``(despiked_df, spike_mask_df)`` — the despiked frame (same shape
        and column order) and the boolean row-mask frame (point columns
        only), for the verification plot.
    """
    selected_channels = np.where(np.asarray(r1_channel_mask, dtype=bool))[0]
    row_of_channel = {int(ch): row for row, ch in enumerate(selected_channels)}
    n_rows = len(r1_df)
    if n_rows != len(selected_channels):
        raise ValueError(
            f"R1 frame has {n_rows} rows but the channel mask selects "
            f"{len(selected_channels)} channels"
        )

    point_cols = _point_columns(r1_df)
    region_masks = masks.get("R1", {})
    despiked = r1_df.copy()
    mask_columns: Dict[int, np.ndarray] = {}
    for col in point_cols:
        row_mask = np.zeros(n_rows, dtype=bool)
        for ch in _in_window_channels(region_masks.get(col), region_windows["R1"]):
            row = row_of_channel.get(int(ch))
            if row is not None:
                row_mask[row] = True
        mask_columns[col] = row_mask
        if row_mask.any():
            despiked[col] = apply_mask_replacement(
                r1_df[col], row_mask, interpolation_method
            )
    spike_mask_df = pd.DataFrame(mask_columns, index=r1_df.index)
    return despiked, spike_mask_df


def apply_masks_to_fluorescence_frame(
    fluor_df: pd.DataFrame,
    masks: MaskSet,
    region_windows: Mapping[str, Tuple[int, int]],
    interpolation_method: str,
) -> pd.DataFrame:
    """Apply the covered-contributor union to the fluorescence frame.

    The frame is the R2+R3 full-plane sum with identity row map (row *i* =
    channel *i*). Covered contributors: R2 in its certified window, R3 in
    its certified window; every other channel has no covered contributor
    and retains legacy never-screened behavior.
    """
    n_rows = len(fluor_df)
    despiked = fluor_df.copy()
    for col in _point_columns(fluor_df):
        row_mask = np.zeros(n_rows, dtype=bool)
        for region in ("R2", "R3"):
            channels = _in_window_channels(
                masks.get(region, {}).get(col), region_windows[region]
            )
            row_mask[channels[channels < n_rows]] = True
        if row_mask.any():
            despiked[col] = apply_mask_replacement(
                fluor_df[col], row_mask, interpolation_method
            )
    return despiked


def apply_stored_mask_to_array(
    intensity: np.ndarray,
    channel_indices: Sequence[int],
    window: Tuple[int, int],
    interpolation_method: str,
) -> np.ndarray:
    """Apply a single region's stored mask to a full-plane intensity array.

    The serving path (web stored-mask toggle, spec §4.6) holds each region's
    spectrum as a 2148-channel numpy array with identity channel→row map.
    This wraps the shared legacy interpolation
    (:func:`~sherloc_pipeline.core.preprocessing.apply_mask_replacement`) so
    the web route despikes exactly the way the pipeline does, with the same
    certified-window restriction: channel indices outside
    ``window`` are ignored.

    Args:
        intensity: Full-plane intensity array (length == plane width).
        channel_indices: Absolute channel indices flagged for this region.
        window: The region's certified detection window ``(lo, hi)``.
        interpolation_method: pandas interpolation method.

    Returns:
        A new ``float`` numpy array with masked channels interpolated. The
        input is returned as a float copy when nothing is in-window.
    """
    n = len(intensity)
    row_mask = np.zeros(n, dtype=bool)
    in_window = _in_window_channels(np.asarray(channel_indices), window)
    in_window = in_window[in_window < n]
    row_mask[in_window] = True
    if not row_mask.any():
        return np.asarray(intensity, dtype=float).copy()
    series = pd.Series(np.asarray(intensity, dtype=float))
    # ``copy=True`` so the result is writable: downstream consumers (e.g.
    # ``stitch_r123_spectrum``'s in-place ``nan_to_num``) mutate their inputs.
    return apply_mask_replacement(series, row_mask, interpolation_method).to_numpy(copy=True)


#: Per-channel contributor segments of the Loupe overlap-summation R123
#: construction, *derived from* the boundary constants in
#: ``core/r123_stitching.py`` (never hardcoded separately — spec §3.3).
#: Each entry is ``(start, end, (regions...))``
#: with ``[start, end)`` half-open channel ranges.
def _r123_summation_segments() -> Sequence[Tuple[int, int, Tuple[str, ...]]]:
    from sherloc_pipeline.core.r123_stitching import (
        _OVERLAP1_END,
        _OVERLAP2_END,
        _R1_ONLY_END,
        _R2_ONLY_END,
        _R3_ONLY_END,
    )

    return (
        (0, _R1_ONLY_END, ("R1",)),
        (_R1_ONLY_END, _OVERLAP1_END, ("R1", "R2")),
        (_OVERLAP1_END, _R2_ONLY_END, ("R2",)),
        (_R2_ONLY_END, _OVERLAP2_END, ("R2", "R3")),
        (_OVERLAP2_END, _R3_ONLY_END, ("R3",)),
    )


def count_uncovered_contributor_channels(
    construction: str,
    region_windows: Mapping[str, Tuple[int, int]],
    n_channels: int = 2148,
) -> int:
    """Count combined-view channels carrying ≥1 uncovered contributor.

    A *contributor* to a channel is a region that supplies that channel
    under the view's construction; it is *uncovered* there iff the channel
    lies outside the region's certified detection window — such a
    contribution is not CR-screened and retains legacy never-screened
    behavior (spec §3.3). The count is the disclosure value
    ``n_uncovered_contributor_channels`` returned by composite web views
    (§4.6); it is **derived** here from the construction's own segment
    definition plus ``region_windows`` and is never a hardcoded constant
    (review F1-R4). With today's certified windows it evaluates to 207 for
    ``r123_summation`` (147 overlap + 60 edge) and 2148 for
    ``fluorescence_sum``; single-region views report 0.

    Args:
        construction: ``"r123_summation"``, ``"fluorescence_sum"``, or
            ``"single_region"``.
        region_windows: Certified detection windows ``{region: (lo, hi)}``
            (typically ``DEFAULT_MANIFEST.region_windows``).
        n_channels: Plane width.

    Returns:
        Number of channels with at least one uncovered contributor.

    Raises:
        ValueError: On an unknown ``construction``.
    """
    def _covered(region: str, channel: int) -> bool:
        lo, hi = region_windows[region]
        return lo <= channel < hi

    if construction == "single_region":
        return 0
    if construction == "fluorescence_sum":
        # R2+R3 full-plane sum: every channel is contributed by both R2 and
        # R3, whose certified windows are disjoint — so at least one
        # contributor is always uncovered.
        segments = ((0, n_channels, ("R2", "R3")),)
    elif construction == "r123_summation":
        segments = _r123_summation_segments()
    else:
        raise ValueError(
            f"unknown construction {construction!r}; expected "
            "'r123_summation', 'fluorescence_sum', or 'single_region'"
        )

    count = 0
    for start, end, regions in segments:
        for channel in range(start, min(end, n_channels)):
            if any(not _covered(r, channel) for r in regions):
                count += 1
    return count


def apply_masks_to_r123_frame(
    r123_df: pd.DataFrame,
    masks: MaskSet,
    winning_region_map: np.ndarray,
    region_windows: Mapping[str, Tuple[int, int]],
    interpolation_method: str,
) -> pd.DataFrame:
    """Apply the winning-region rule to the masked-assignment R123 frame.

    Row *i* = channel *i*. A channel changes iff the region that won the
    assignment at that channel flags it; a flagged non-winning region
    produces no change.
    """
    n_rows = len(r123_df)
    winner = np.asarray(winning_region_map, dtype="<U2")
    if winner.shape[0] != n_rows:
        raise ValueError(
            f"winning_region_map length {winner.shape[0]} != frame rows {n_rows}"
        )
    despiked = r123_df.copy()
    for col in _point_columns(r123_df):
        row_mask = np.zeros(n_rows, dtype=bool)
        for region in REGION_ASSIGNMENT_ORDER:
            channels = _in_window_channels(
                masks.get(region, {}).get(col), region_windows[region]
            )
            channels = channels[channels < n_rows]
            row_mask[channels] |= winner[channels] == region
        if row_mask.any():
            despiked[col] = apply_mask_replacement(
                r123_df[col], row_mask, interpolation_method
            )
    return despiked
