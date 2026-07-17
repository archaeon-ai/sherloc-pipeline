"""Synthetic tests for the optional atomic delivery handoff."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
from pathlib import Path

import pytest
from PIL import Image

from sherloc_pipeline.services import handoff
from sherloc_pipeline.services.errors import HandoffError
from sherloc_pipeline.services.handoff import HandoffService
from sherloc_pipeline.services.pipeline import PipelineService


PRODUCT = "SC2_1806_0827295848_123ECM_N0870000SRLC11470_0000LMJ01"
RUN_ID = "run-1806"


def _write_png(path: Path, size: tuple[int, int], color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=color).save(path)


def _write_vicar(
    path: Path, *, size: tuple[int, int] = (1648, 1200), bands: int = 1
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    label_size = 512
    label = (
        f"LBLSIZE={label_size} FORMAT='BYTE' NL={height} NS={width} NB={bands} "
    ).encode("ascii")
    path.write_bytes(
        label.ljust(label_size, b" ")
        + bytes([100]) * (width * height * bands)
    )


def _write_pds3(path: Path, *, embedded_vicar_label: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record_bytes = 1648
    image_record = 3
    header = (
        "PDS_VERSION_ID = PDS3\r\n"
        f"RECORD_BYTES = {record_bytes}\r\n"
        f"^IMAGE = {image_record}\r\n"
        "LINES = 1200\r\n"
        "LINE_SAMPLES = 1648\r\n"
        "SAMPLE_BITS = 8\r\n"
        + (
            'LBLSIZE = "16480 FORMAT=\'BYTE\' NL=1200 NS=1648"\r\n'
            if embedded_vicar_label
            else ""
        )
        +
        "END\r\n"
    ).encode("ascii")
    offset = (image_record - 1) * record_bytes
    path.write_bytes(
        header.ljust(offset, b" ") + bytes([100]) * (1648 * 1200)
    )


def _workspace(source_root: Path, *, colorized: bool = True) -> Path:
    working = source_root / "sol_1806" / "detail" / "workspace"
    _write_png(working / "img" / f"{PRODUCT}.PNG", (1648, 1200), 100)
    working.joinpath("img", f"{PRODUCT}.CSV").write_text("image metadata\n")
    if colorized:
        color = source_root / "sol_1806_colorized" / "detail" / "workspace"
        _write_png(color / "img" / f"{PRODUCT}.PNG", (800, 600), 150)
        color.joinpath("img", f"{PRODUCT}.CSV").write_text("image metadata\n")
        color.joinpath("spatial.csv").write_text("x,y\n")
        color.joinpath("loupe.csv").write_text("points,1\n")
    return working


def test_unconfigured_handoff_is_an_explicit_skip(tmp_path: Path) -> None:
    result = HandoffService().publish_if_configured(
        output_dir=None,
        run_id=RUN_ID,
        source_root=tmp_path / "absent",
        working_dir=None,
    )
    assert result.artifacts == []
    assert result.metadata == {"status": "skipped", "reason": "not_configured"}


def test_pipeline_publishes_only_after_summary_and_before_success() -> None:
    source = inspect.getsource(PipelineService.run_full_pipeline)
    assert source.index('capture_stage("summary")') < source.index(
        'os.environ.get("SHERLOC_HANDOFF_DIR")'
    )
    assert source.index('os.environ.get("SHERLOC_HANDOFF_DIR")') < source.index(
        "Full pipeline completed successfully."
    )
    assert "resolve_scan_context(" in source
    assert "ingestion.find_working_directory(sol, scan)" not in source


def test_configured_handoff_publishes_closed_evidence_atomically(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    output = tmp_path / "handoff"

    result = HandoffService().publish_if_configured(
        output_dir=output,
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )

    ready = output / f"{RUN_ID}.ready"
    assert result.artifacts == [ready]
    assert result.metadata["status"] == "ready"
    assert not (output / f"{RUN_ID}.tmp").exists()
    document = json.loads(ready.read_text())
    assert set(document) == {
        "schema_version", "producer", "run_id", "created_at", "epoch",
        "selector", "entries",
    }
    assert document["schema_version"] == "aci-handoff-manifest.v2"
    assert document["selector"] == {"product_ids": [PRODUCT]}
    assert len(document["entries"]) == 2
    raw = next(item for item in document["entries"] if item["role"] == "raw_grayscale")
    color = next(item for item in document["entries"] if item["role"] == "colorized")
    assert raw["rendition_key"] == "product"
    assert raw["source_rel_locator"] == f"sol_1806/detail/workspace/img/{PRODUCT}.PNG"
    assert color["rendition_key"] == color["source_rel_locator"]
    assert color["coordinate_frame"] == "workspace_crop"
    assert {item["role"] for item in color["sidecars"]} == {"spatial", "loupe"}
    for entry in document["entries"]:
        source_path = source.joinpath(*Path(entry["source_rel_locator"]).parts)
        assert entry["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()


def test_rgb_workspace_falls_back_to_full_frame_vicar_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    Image.new("RGB", (1648, 1200), color=(1, 2, 3)).save(
        working / "img" / f"{PRODUCT}.PNG"
    )
    vicar = working.parent / f"{PRODUCT}.IMG"
    _write_vicar(vicar)

    result = HandoffService().publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )

    raw = json.loads(result.artifacts[0].read_text())["entries"][0]
    assert raw["source_rel_locator"] == f"sol_1806/detail/{PRODUCT}.IMG"
    assert raw["source_format"] == "vicar_img"
    assert raw["source_sha256"] == hashlib.sha256(vicar.read_bytes()).hexdigest()
    assert raw["sha256"] == hashlib.sha256(
        handoff._canonical_png_bytes(vicar)
    ).hexdigest()


def test_img_only_workspace_publishes_full_frame_vicar_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = source / "sol_1806" / "detail" / "workspace"
    (working / "img").mkdir(parents=True)
    vicar = working.parent / f"{PRODUCT}.IMG"
    _write_vicar(vicar)

    result = HandoffService().publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )

    raw = json.loads(result.artifacts[0].read_text())["entries"][0]
    assert raw["source_format"] == "vicar_img"
    assert (raw["width_px"], raw["height_px"]) == (1648, 1200)


def test_mixed_workspace_includes_img_only_sibling_product(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    sibling = "SC2_1806_0000000000_000ECM_N0000000SRLC00000_0000LMJ01"
    _write_vicar(working.parent / f"{sibling}.IMG")

    result = HandoffService().publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )

    document = json.loads(result.artifacts[0].read_text())
    assert document["selector"] == {"product_ids": sorted([PRODUCT, sibling])}
    assert {
        entry["product_id"] for entry in document["entries"]
        if entry["role"] == "raw_grayscale"
    } == {PRODUCT, sibling}


def test_multiband_vicar_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / f"{PRODUCT}.IMG"
    _write_vicar(source, bands=3)

    with pytest.raises(HandoffError, match="not single-band"):
        handoff._canonical_png_bytes(source)


def test_single_band_pds3_without_bands_field_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / f"{PRODUCT}.IMG"
    _write_pds3(source)

    assert handoff._canonical_png_bytes(source).startswith(b"\x89PNG\r\n\x1a\n")


def test_pds3_with_embedded_vicar_label_is_classified_as_pds3(
    tmp_path: Path,
) -> None:
    source = tmp_path / f"{PRODUCT}.IMG"
    _write_pds3(source, embedded_vicar_label=True)

    assert handoff._canonical_png_bytes(source).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    "label",
    [
        b"LBLSIZE=512 FORMAT=HALF NL=1200 NS=1648 NB=1 ",
        (
            b"PDS_VERSION_ID=PDS3\r\nRECORD_BYTES=1648\r\n^IMAGE=3\r\n"
            b"LINES=1200\r\nLINE_SAMPLES=1648\r\nSAMPLE_BITS=16\r\n"
            b"BANDS=1\r\nEND\r\n"
        ),
    ],
)
def test_non_byte_vicar_source_is_rejected(tmp_path: Path, label: bytes) -> None:
    source = tmp_path / f"{PRODUCT}.IMG"
    source.write_bytes(label.ljust(512, b" ") + bytes(1648 * 1200 * 2))

    with pytest.raises(HandoffError, match="not byte-encoded"):
        handoff._canonical_png_bytes(source)


def test_raw_only_workspace_is_valid(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    result = HandoffService().publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )
    document = json.loads(result.artifacts[0].read_text())
    assert [entry["role"] for entry in document["entries"]] == ["raw_grayscale"]


@pytest.mark.parametrize(
    ("mode", "size"),
    [
        ("RGB", (1648, 1200)),
        ("RGB", (1544, 1156)),
        ("L", (1544, 1156)),
    ],
)
def test_noncanonical_raw_image_fails_closed(
    tmp_path: Path,
    mode: str,
    size: tuple[int, int],
) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    color: int | tuple[int, int, int] = 1 if mode == "L" else (1, 2, 3)
    Image.new(mode, size, color=color).save(
        working / "img" / f"{PRODUCT}.PNG"
    )
    with pytest.raises(HandoffError, match="full frame"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_renamed_non_png_image_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    Image.new("L", (1648, 1200), color=100).save(
        working / "img" / f"{PRODUCT}.PNG",
        format="JPEG",
    )
    with pytest.raises(HandoffError, match="valid PNG"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_lowercase_png_product_name_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    uppercase = working / "img" / f"{PRODUCT}.PNG"
    lowercase = uppercase.with_suffix(".png")
    uppercase.rename(lowercase)

    with pytest.raises(HandoffError, match="invalid product identity"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_auxiliary_range_pngs_are_not_handoff_products(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    auxiliary = working / "img" / f"{PRODUCT}_145-185.png"
    Image.new("RGB", (320, 240), color=(1, 2, 3)).save(auxiliary)

    result = HandoffService().publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )

    document = json.loads(result.artifacts[0].read_text())
    assert document["selector"] == {"product_ids": [PRODUCT]}
    assert all(item["product_id"] == PRODUCT for item in document["entries"])


def test_aci_product_without_same_stem_metadata_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    other = "SC2_1806_0000000000_000ECM_N0000000SRLC00000_0000LMJ01"
    _write_png(working / "img" / f"{other}.PNG", (1648, 1200), 200)

    with pytest.raises(HandoffError, match="missing same-stem metadata"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_same_stem_metadata_outside_data_root_fails_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    metadata = working / "img" / f"{PRODUCT}.CSV"
    outside = tmp_path / "outside.CSV"
    metadata.replace(outside)
    metadata.symlink_to(outside)

    with pytest.raises(HandoffError, match="escapes"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_watson_png_with_metadata_is_not_an_aci_handoff_product(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    watson = "SI1_1806_0827295848_123ECM_N0870000SRLC11470_0000LMJ01"
    Image.new("RGB", (1600, 1200), color=(1, 2, 3)).save(
        working / "img" / f"{watson}.PNG"
    )
    working.joinpath("img", f"{watson}.CSV").write_text("image metadata\n")

    result = HandoffService().publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )

    document = json.loads(result.artifacts[0].read_text())
    assert document["selector"] == {"product_ids": [PRODUCT]}
    assert all(item["product_id"] == PRODUCT for item in document["entries"])


def test_configured_invalid_destination_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    destination = tmp_path / "not-a-directory"
    destination.write_text("occupied")
    with pytest.raises(HandoffError, match="could not be published"):
        HandoffService().publish_if_configured(
            output_dir=destination,
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_existing_ready_manifest_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    service = HandoffService()
    first = service.publish_if_configured(
        output_dir=tmp_path / "handoff",
        run_id=RUN_ID,
        source_root=source,
        working_dir=working,
    )
    before = first.artifacts[0].read_bytes()
    with pytest.raises(HandoffError, match="could not be published"):
        service.publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )
    assert first.artifacts[0].read_bytes() == before


def test_existing_temporary_manifest_is_never_removed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    output = tmp_path / "handoff"
    output.mkdir()
    temporary = output / f"{RUN_ID}.tmp"
    temporary.write_text("competing publisher\n")

    with pytest.raises(HandoffError, match="could not be published"):
        HandoffService().publish_if_configured(
            output_dir=output,
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )

    assert temporary.read_text() == "competing publisher\n"
    assert not (output / f"{RUN_ID}.ready").exists()


def test_ready_created_during_publish_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    output = tmp_path / "handoff"
    ready = output / f"{RUN_ID}.ready"
    original_link = os.link

    def competing_link(source_path: Path, destination_path: Path) -> None:
        ready.write_text("competing publisher\n")
        original_link(source_path, destination_path)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(HandoffError, match="could not be published"):
        HandoffService().publish_if_configured(
            output_dir=output,
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )
    assert ready.read_text() == "competing publisher\n"
    assert not (output / f"{RUN_ID}.tmp").exists()


def test_post_link_directory_fsync_failure_removes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    output = tmp_path / "handoff"
    original_fsync = os.fsync
    directory_calls = 0

    def fail_second_directory_fsync(fd: int) -> None:
        nonlocal directory_calls
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_calls += 1
            if directory_calls == 2:
                raise OSError("injected directory durability failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_second_directory_fsync)
    with pytest.raises(HandoffError, match="could not be published"):
        HandoffService().publish_if_configured(
            output_dir=output,
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )

    assert directory_calls >= 2
    assert not (output / f"{RUN_ID}.ready").exists()
    assert not (output / f"{RUN_ID}.tmp").exists()


def test_source_outside_data_root_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(tmp_path / "outside", colorized=False)
    with pytest.raises(HandoffError, match="escapes"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_symlinked_source_outside_data_root_is_rejected_before_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    product = working / "img" / f"{PRODUCT}.PNG"
    outside = tmp_path / "outside.PNG"
    product.replace(outside)
    product.symlink_to(outside)
    hash_calls = 0

    def record_hash(path: Path) -> str:
        nonlocal hash_calls
        hash_calls += 1
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(handoff, "_sha256", record_hash)
    with pytest.raises(HandoffError, match="escapes"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )

    assert hash_calls == 0


def test_symlinked_image_directory_outside_data_root_is_rejected_before_listing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    working = source / "sol_1806" / "detail" / "workspace"
    working.mkdir(parents=True)
    outside = tmp_path / "outside-img"
    outside.mkdir()
    (working / "img").symlink_to(outside, target_is_directory=True)

    with pytest.raises(HandoffError, match="escapes"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_colorized_product_without_raw_counterpart_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    other = "SC2_1806_0000000000_000ECM_N0000000SRLC00000_0000LMJ01"
    color = source / "sol_1806_colorized" / "detail" / "workspace"
    _write_png(color / "img" / f"{other}.PNG", (640, 480), 200)
    color.joinpath("img", f"{other}.CSV").write_text("image metadata\n")
    with pytest.raises(HandoffError, match="no raw product"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_existing_empty_colorized_workspace_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    (source / "sol_1806_colorized/detail/workspace/img").mkdir(parents=True)
    with pytest.raises(HandoffError, match="no product PNG images"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )


def test_missing_colorized_sidecar_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source)
    (source / "sol_1806_colorized/detail/workspace/spatial.csv").unlink()
    with pytest.raises(HandoffError, match="cannot be read"):
        HandoffService().publish_if_configured(
            output_dir=tmp_path / "handoff",
            run_id=RUN_ID,
            source_root=source,
            working_dir=working,
        )
