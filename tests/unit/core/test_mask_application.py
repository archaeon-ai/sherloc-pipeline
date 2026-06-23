"""Mask application semantics on the three pipeline representations.

Covers the contributor-set principle and certified-window applicability
(spec §3.3, MLD-SYS-014/015) plus the R123 stitch-alignment test: the
derived winning-region map must agree, channel by channel, with what
``create_r123_spectrum`` actually assigns (MLD-SYS-014 AC1 — the map is
derived from the construction's definition, never hardcoded separately).
"""

import numpy as np
import pandas as pd
import pytest

from sherloc_pipeline.core.mask_application import (
    apply_masks_to_fluorescence_frame,
    apply_masks_to_r1_frame,
    apply_masks_to_r123_frame,
    apply_stored_mask_to_array,
    count_uncovered_contributor_channels,
    derive_region_channel_masks,
    derive_winning_region_map,
)

N_CHANNELS = 2148

#: Certified detection windows — pinned literals here (the production code
#: receives them from the frozen manifest; tests assert against the
#: manifest in test_manifest_windows_match_pin below).
WINDOWS = {"R1": (52, 575), "R2": (575, 1677), "R3": (1677, 2140)}


def test_manifest_windows_match_pin():
    from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

    assert dict(DEFAULT_MANIFEST.region_windows) == WINDOWS


@pytest.fixture(scope="module")
def config():
    from sherloc_pipeline.config import get_config

    return get_config()


@pytest.fixture(scope="module")
def channel_masks(config):
    _, masks = derive_region_channel_masks(config)
    return masks


def _frame(n_rows: int, n_points: int = 3, axis_name: str = "wavelength",
           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {axis_name: np.linspace(100.0, 400.0, n_rows)}
    for p in range(n_points):
        data[p] = 100.0 + rng.standard_normal(n_rows)
    return pd.DataFrame(data)


class TestR123StitchAlignment:
    """Derived winning map vs the actual create_r123_spectrum assignment."""

    def test_winning_map_matches_construction(self, config, channel_masks):
        from sherloc_pipeline.core.data_ingestion import DataIngestion

        n_spectra = 2
        # Synthetic dark_sub_df with the Loupe section layout: R1 rows hold
        # 1.0, R2 rows 2.0, R3 rows 3.0; the stand-in section header rows at
        # iloc n and 2n+1 are skipped by the construction's slicing.
        dark_sub = pd.DataFrame(
            [[1.0] * N_CHANNELS] * n_spectra
            + [[np.nan] * N_CHANNELS]
            + [[2.0] * N_CHANNELS] * n_spectra
            + [[np.nan] * N_CHANNELS]
            + [[3.0] * N_CHANNELS] * n_spectra
        )

        ingestion = DataIngestion(
            base_data_dir="tests/fixtures/loupe", results_dir="/tmp/unused"
        )
        r123 = ingestion.create_r123_spectrum(dark_sub, n_spectra)

        winner = derive_winning_region_map(channel_masks)
        value_of = {"": 0.0, "R1": 1.0, "R2": 2.0, "R3": 3.0}
        expected = np.array([value_of[w] for w in winner])
        np.testing.assert_array_equal(r123[0].to_numpy(), expected)

    def test_every_assigned_channel_has_unique_winner(self, channel_masks):
        winner = derive_winning_region_map(channel_masks)
        # Last-writer-wins order: any channel claimed by multiple regions
        # must resolve to the later region in (R1, R2, R3).
        for ch in range(N_CHANNELS):
            claimers = [
                r for r in ("R1", "R2", "R3") if channel_masks[r][ch]
            ]
            if claimers:
                assert winner[ch] == claimers[-1]
            else:
                assert winner[ch] == ""


class TestR123Application:
    """Winning-region rule with a hand-built map containing real overlap."""

    def _setup(self):
        n_rows = 40
        frame = _frame(n_rows, n_points=2, seed=1)
        # Hand-built winning map: rows 0-19 win R1, 20-29 win R2 (overlap
        # zone where R1 also claims), 30-39 win R3.
        winner = np.array(
            ["R1"] * 20 + ["R2"] * 10 + ["R3"] * 10, dtype="<U2"
        )
        windows = {"R1": (0, 30), "R2": (15, 35), "R3": (25, 40)}
        return frame, winner, windows

    def test_winning_region_flag_changes_channel(self):
        frame, winner, windows = self._setup()
        masks = {"R2": {0: np.array([22])}}  # R2 wins channel 22
        out = apply_masks_to_r123_frame(frame, masks, winner, windows, "linear")
        assert out[0].iloc[22] != frame[0].iloc[22]
        # Only that channel changed, only that point
        changed = out[0].to_numpy() != frame[0].to_numpy()
        assert list(np.where(changed)[0]) == [22]
        np.testing.assert_array_equal(out[1].to_numpy(), frame[1].to_numpy())

    def test_non_winning_region_flag_is_inert(self):
        frame, winner, windows = self._setup()
        # Channel 22 is inside R1's window (0,30) but R2 wins it: an R1
        # flag there must produce no change (MLD-SYS-014 AC1).
        masks = {"R1": {0: np.array([22])}}
        out = apply_masks_to_r123_frame(frame, masks, winner, windows, "linear")
        np.testing.assert_array_equal(out[0].to_numpy(), frame[0].to_numpy())

    def test_out_of_window_flag_is_inert(self):
        frame, winner, windows = self._setup()
        # Channel 36 wins R3 and is inside R3's window; channel 5 wins R1.
        # An R3 flag at channel 5 is both out-of-window and non-winning.
        masks = {"R3": {0: np.array([5])}}
        out = apply_masks_to_r123_frame(frame, masks, winner, windows, "linear")
        np.testing.assert_array_equal(out[0].to_numpy(), frame[0].to_numpy())

    def test_replacement_matches_shared_helper(self):
        from sherloc_pipeline.core.preprocessing import apply_mask_replacement

        frame, winner, windows = self._setup()
        masks = {"R3": {1: np.array([33, 34])}}
        out = apply_masks_to_r123_frame(frame, masks, winner, windows, "linear")
        row_mask = np.zeros(len(frame), dtype=bool)
        row_mask[[33, 34]] = True
        expected = apply_mask_replacement(frame[1], row_mask, "linear")
        pd.testing.assert_series_equal(out[1], expected, check_names=False)


class TestFluorescenceUnion:
    """Covered-contributor union: R2 in [575,1677), R3 in [1677,2140)."""

    def test_union_and_boundaries(self):
        frame = _frame(N_CHANNELS, n_points=1, seed=2)
        masks = {
            # R2 flags: 574 (below window — inert), 575, 1676 (covered),
            # 1677 (above window — inert)
            "R2": {0: np.array([574, 575, 1676, 1677])},
            # R3 flags: 1676 (below window — inert), 1677, 2139 (covered),
            # plus an absurd 2140 (above window — inert)
            "R3": {0: np.array([1676, 1677, 2139, 2140])},
        }
        out = apply_masks_to_fluorescence_frame(frame, masks, WINDOWS, "linear")
        changed = set(np.where(out[0].to_numpy() != frame[0].to_numpy())[0])
        assert changed == {575, 1676, 1677, 2139}

    def test_edge_segments_never_screened(self):
        """Channels [0,52) and [2140,2148) have no covered contributor —
        flags there from any region are inert (MLD-SYS-015 AC1)."""
        frame = _frame(N_CHANNELS, n_points=1, seed=3)
        masks = {
            "R2": {0: np.array([0, 10, 51, 2140, 2147])},
            "R3": {0: np.array([0, 10, 51, 2140, 2147])},
        }
        out = apply_masks_to_fluorescence_frame(frame, masks, WINDOWS, "linear")
        np.testing.assert_array_equal(out[0].to_numpy(), frame[0].to_numpy())

    def test_no_masks_is_identity(self):
        frame = _frame(N_CHANNELS, n_points=2, seed=4)
        out = apply_masks_to_fluorescence_frame(frame, {}, WINDOWS, "linear")
        pd.testing.assert_frame_equal(out, frame)


class TestR1FrameMapping:
    def test_channel_to_row_mapping(self, channel_masks):
        r1_mask = channel_masks["R1"]
        selected = np.where(r1_mask)[0]
        n_rows = len(selected)
        frame = _frame(n_rows, n_points=2, axis_name="raman_shift", seed=5)

        # Flag the 9th selected channel of point 0 — must hit row 8.
        target_channel = int(selected[8])
        masks = {"R1": {0: np.array([target_channel])}}
        out, spike_df = apply_masks_to_r1_frame(
            frame, masks, r1_mask, WINDOWS, "linear"
        )
        changed = np.where(out[0].to_numpy() != frame[0].to_numpy())[0]
        np.testing.assert_array_equal(changed, [8])
        assert spike_df[0].iloc[8]
        assert spike_df[0].sum() == 1
        assert not spike_df[1].any()

    def test_out_of_frame_channel_skipped(self, channel_masks):
        """Flags outside the exported R1 window are persisted upstream but
        not applicable to this frame — silently skipped (spec §3.3)."""
        r1_mask = channel_masks["R1"]
        n_rows = int(r1_mask.sum())
        frame = _frame(n_rows, n_points=1, axis_name="raman_shift", seed=6)
        # Channel 5 is outside both the R1 selection and the R1 window;
        # channel 2000 is outside the R1 window entirely.
        masks = {"R1": {0: np.array([5, 2000])}}
        out, spike_df = apply_masks_to_r1_frame(
            frame, masks, r1_mask, WINDOWS, "linear"
        )
        np.testing.assert_array_equal(out[0].to_numpy(), frame[0].to_numpy())
        assert not spike_df[0].any()

    def test_row_count_mismatch_raises(self, channel_masks):
        frame = _frame(10, n_points=1, axis_name="raman_shift")
        with pytest.raises(ValueError, match="rows"):
            apply_masks_to_r1_frame(
                frame, {}, channel_masks["R1"], WINDOWS, "linear"
            )


class TestUncoveredContributorCount:
    """Disclosure count for composite views (spec §3.3, review F1-R4).

    The count is *derived* from the construction's segment map plus the
    certified windows — never a hardcoded constant. Today's certified
    windows pin it to 207 (R123 summation) and 2148 (fluorescence sum);
    those literals are change-detectors here, recomputed from WINDOWS.
    """

    def test_r123_summation_is_207(self):
        assert count_uncovered_contributor_channels("r123_summation", WINDOWS) == 207

    def test_fluorescence_sum_is_all_channels(self):
        assert (
            count_uncovered_contributor_channels("fluorescence_sum", WINDOWS)
            == N_CHANNELS
        )

    def test_single_region_is_zero(self):
        assert count_uncovered_contributor_channels("single_region", WINDOWS) == 0

    def test_matches_manifest_derivation(self):
        """The route derives the count from the frozen manifest windows;
        the manifest-sourced value must equal the WINDOWS-pinned value."""
        from sherloc_pipeline.ml_despike.manifest import DEFAULT_MANIFEST

        rw = DEFAULT_MANIFEST.region_windows
        assert count_uncovered_contributor_channels("r123_summation", rw) == 207
        assert count_uncovered_contributor_channels("fluorescence_sum", rw) == N_CHANNELS

    def test_207_decomposes_to_147_overlap_plus_60_edge(self):
        """Independent reconstruction from the §3.3 table: the count is the
        sum over channels carrying ≥1 uncovered contributor."""
        # Overlap-1 [565,690): 125 ch (R2<575 below; R1≥575 above).
        # Overlap-2 [1668,1690): 22 ch. Edge [0,52): 52; [2140,2148): 8.
        assert 125 + 22 == 147
        assert 52 + 8 == 60
        assert count_uncovered_contributor_channels("r123_summation", WINDOWS) == 147 + 60

    def test_unknown_construction_raises(self):
        with pytest.raises(ValueError, match="unknown construction"):
            count_uncovered_contributor_channels("bogus", WINDOWS)


class TestApplyStoredMaskToArray:
    """Full-plane stored-mask application used by the web serving path."""

    def test_replaces_in_window_channel(self):
        rng = np.random.default_rng(11)
        arr = rng.normal(500.0, 5.0, size=N_CHANNELS)
        out = apply_stored_mask_to_array(arr, [800], WINDOWS["R2"], "linear")
        changed = np.where(out != arr)[0]
        np.testing.assert_array_equal(changed, [800])
        # Linear interpolation = mean of the two neighbours.
        assert out[800] == pytest.approx((arr[799] + arr[801]) / 2.0)

    def test_out_of_window_channel_ignored(self):
        rng = np.random.default_rng(12)
        arr = rng.normal(500.0, 5.0, size=N_CHANNELS)
        # 100 is below the R2 window [575,1677): not screenable for R2.
        out = apply_stored_mask_to_array(arr, [100], WINDOWS["R2"], "linear")
        np.testing.assert_array_equal(out, arr.astype(float))

    def test_empty_mask_is_float_copy(self):
        arr = np.arange(N_CHANNELS, dtype=np.int64)
        out = apply_stored_mask_to_array(arr, [], WINDOWS["R1"], "linear")
        assert out.dtype == float
        np.testing.assert_array_equal(out, arr.astype(float))
        assert out is not arr
