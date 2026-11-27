"""download_aci_image refuses non-image bodies and replaces non-image cache files.

An archive error page served with HTTP 200 was once stored under 392 ``.IMG``
names in the PDS cache and then kept forever by the existing-file skip.
"""

from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")

from sherloc_pipeline.core.pds_client import PDSDownloader  # noqa: E402

VICAR_BODY = b"LBLSIZE=2048            FORMAT='BYTE'" + b"\0" * 2048
PDS3_BODY = b"ODL_VERSION_ID                 = ODL3\r\n" + b"\0" * 512
HTML_BODY = b"<!DOCTYPE html>\n<html><head><title>Not Found</title></head><body></body></html>\n"
URL = "https://example.invalid/data/SC0_0004_0667297936_035ECM_N0010052SRLC10002_0000LUJ01.IMG"


def _downloader(tmp_path: Path, handler) -> tuple[PDSDownloader, list]:
    calls: list = []

    def recording(request):
        calls.append(request.url)
        return handler(request)

    dl = PDSDownloader(cache_dir=tmp_path)
    dl._client = httpx.Client(transport=httpx.MockTransport(recording))
    return dl, calls


def _serve(body: bytes, content_type: str = "application/octet-stream", **headers):
    def handler(request):
        return httpx.Response(200, content=body, headers={"content-type": content_type, **headers})

    return handler


@pytest.mark.parametrize("body", [VICAR_BODY, PDS3_BODY])
def test_image_body_is_written(tmp_path, body):
    dl, _ = _downloader(tmp_path, _serve(body))
    dest = tmp_path / "sol_0004" / "data_aci" / "a.IMG"
    assert dl.download_aci_image(URL, dest) is True
    assert dest.read_bytes() == body


@pytest.mark.parametrize("content_type", ["text/html; charset=utf-8", "application/octet-stream", ""])
def test_html_body_is_refused(tmp_path, content_type):
    dl, _ = _downloader(tmp_path, _serve(HTML_BODY, content_type))
    dest = tmp_path / "a.IMG"
    assert dl.download_aci_image(URL, dest) is False
    assert not dest.exists()
    assert not dest.with_suffix(".IMG.tmp").exists()


def test_text_content_type_is_refused_even_with_image_bytes(tmp_path):
    dl, _ = _downloader(tmp_path, _serve(VICAR_BODY, "text/plain"))
    dest = tmp_path / "a.IMG"
    assert dl.download_aci_image(URL, dest) is False
    assert not dest.exists()


def test_short_body_is_refused(tmp_path):
    def handler(request):
        resp = httpx.Response(200, content=VICAR_BODY, headers={"content-type": "application/octet-stream"})
        resp.headers["content-length"] = str(len(VICAR_BODY) + 10)
        return resp

    dl, _ = _downloader(tmp_path, handler)
    dest = tmp_path / "a.IMG"
    assert dl.download_aci_image(URL, dest) is False
    assert not dest.exists()


def test_existing_html_file_is_fetched_again(tmp_path):
    dest = tmp_path / "a.IMG"
    dest.write_bytes(HTML_BODY)
    dl, calls = _downloader(tmp_path, _serve(VICAR_BODY))
    assert dl.download_aci_image(URL, dest) is True
    assert len(calls) == 1
    assert dest.read_bytes() == VICAR_BODY


def test_existing_html_file_is_removed_when_refetch_fails(tmp_path):
    dest = tmp_path / "a.IMG"
    dest.write_bytes(HTML_BODY)
    dl, _ = _downloader(tmp_path, _serve(HTML_BODY, "text/html"))
    assert dl.download_aci_image(URL, dest) is False
    assert not dest.exists()


def test_existing_image_file_is_kept_without_a_request(tmp_path):
    dest = tmp_path / "a.IMG"
    dest.write_bytes(VICAR_BODY)
    dl, calls = _downloader(tmp_path, _serve(HTML_BODY, "text/html"))
    assert dl.download_aci_image(URL, dest) is True
    assert calls == []
    assert dest.read_bytes() == VICAR_BODY
