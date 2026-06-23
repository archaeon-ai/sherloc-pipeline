#!/usr/bin/env python3
"""Release-readiness gate: live-fetch the published model artifact and verify.

The published model-artifact URL is a **PR-blocking**
release-readiness gate: before merge, the URL baked into
``ModelManifest.download_url`` must be live-fetched and its bytes verified
against the pinned ``ModelManifest.sha256``. This script *is* that gate, in
runnable, auditable form — maintainers run it after publishing the release
(``model-cr-despike-1.3``), and it records the JSON verification record.

It exercises the real production fetch path (``ml_despike.resolve_artifact``),
which streams the download, hashes it, and only accepts the file if the digest
matches the pinned value (a host-side asset swap cannot pass). By default it
fetches into a throwaway temp directory so the result reflects a clean
live-fetch rather than a warm cache.

Network is required (this is the live gate). CI does NOT run this — the fetch
logic is covered there with a mocked transport
(``tests/unit/ml_despike/`` artifact tests). Run locally only.

Exit codes
----------
0 — the published URL served bytes matching the pinned digest (PASS)
1 — digest mismatch, fetch failure, or any runtime error (FAIL / gate blocks)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from sherloc_pipeline.ml_despike import (
    DEFAULT_MANIFEST,
    ArtifactDigestError,
    ArtifactFetchError,
    resolve_artifact,
)
from sherloc_pipeline.ml_despike.artifact import _sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Release-readiness gate: live-fetch the published ML CR model "
            "artifact and verify it against the pinned sha256."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help=(
            "Verify a local artifact file instead of fetching the published "
            "URL (still digest-checked; use to pre-verify before publishing)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for the fetch (default: a throwaway temp dir = clean live-fetch).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write a JSON verification record.",
    )
    args = parser.parse_args()

    manifest = DEFAULT_MANIFEST
    print(
        "Release-readiness gate (MLD-IFC-006):\n"
        f"  artifact : {manifest.artifact_filename}\n"
        f"  url      : {manifest.download_url}\n"
        f"  expected : {manifest.sha256}\n"
    )

    tmp_ctx = None
    cache_dir = args.cache_dir
    if cache_dir is None and args.artifact is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="release-verify-")
        cache_dir = Path(tmp_ctx.name)

    verdict = "FAIL"
    actual = None
    error = None
    try:
        resolved = resolve_artifact(manifest, artifact_path=args.artifact, cache_dir=cache_dir)
        actual = _sha256_file(resolved)
        if actual == manifest.sha256:
            verdict = "PASS"
            print(f"  resolved : {resolved}")
            print(f"  actual   : {actual}")
            print("\nMLD-IFC-006 release-readiness → PASS (published bytes match the pinned digest)")
        else:
            error = f"digest mismatch: expected {manifest.sha256}, got {actual}"
            print(f"\nMLD-IFC-006 release-readiness → FAIL ({error})", file=sys.stderr)
    except ArtifactDigestError as exc:
        error = f"digest verification failed: {exc}"
        print(f"\nMLD-IFC-006 release-readiness → FAIL ({error})", file=sys.stderr)
    except ArtifactFetchError as exc:
        error = f"fetch failed: {exc}"
        print(f"\nMLD-IFC-006 release-readiness → FAIL ({error})", file=sys.stderr)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if args.out is not None:
        record = {
            "requirement": "MLD-IFC-006",
            "gate": "release-readiness live-fetch digest verification",
            "artifact_filename": manifest.artifact_filename,
            "download_url": manifest.download_url,
            "expected_sha256": manifest.sha256,
            "actual_sha256": actual,
            "verdict": verdict,
            "error": error,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
        print(f"\nverification record written to: {args.out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
