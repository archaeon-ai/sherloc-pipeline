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

from sherloc_pipeline.services.errors import HandoffError
from sherloc_pipeline.services.handoff import HandoffService
from sherloc_pipeline.services.pipeline import PipelineService


PRODUCT = "SC2_1806_0827295848_123ECM_N0870000SRLC11470_0000LMJ01"
RUN_ID = "run-1806"


def _write_png(path: Path, size: tuple[int, int], color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=color).save(path)


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
    assert document["schema_version"] == "aci-handoff-manifest.v1"
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
        assert entry["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()


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


def test_rgb_raw_image_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    working = _workspace(source, colorized=False)
    Image.new("RGB", (1648, 1200), color=(1, 2, 3)).save(
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
