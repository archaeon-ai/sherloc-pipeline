"""Invented release records for public boundary tests."""

import json

from sherloc_pipeline.services.public_release import PublicReleaseIndex


def synthetic_index(tmp_path, monkeypatch, product_ids=()):
    path = tmp_path / "synthetic-release.json"
    path.write_text(json.dumps({
        "schema_version": "public-release-1", "release_id": "synthetic-release",
        "products": [{"product_id": product, "version_id": "synthetic-v1",
                      "sha256": "a" * 64, "released": True} for product in product_ids],
    }))
    index = PublicReleaseIndex(path)
    monkeypatch.setenv("SHERLOC_FEATURE_POLICY", json.dumps({"public_release": {
        "index_path": str(path), "index_sha256": index.sha256,
    }}))
    return index
