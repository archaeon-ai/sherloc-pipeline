"""Contract test: deploy/docker-compose.example.yml is sanitized and parseable.

Pins DEPLOYMENT_CONTRACT.md §3 + §11 against the static example file.

The example MUST be generic — no operator-local paths, no platform
service names, no real credentials. Comments may reference downstream
platforms generically; this test inspects the *parsed* YAML (comments
stripped) for forbidden strings.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_EXAMPLE = REPO_ROOT / "deploy" / "docker-compose.example.yml"

# Strings that MUST NOT appear in parsed config values. Operator-local
# paths + downstream-platform names are the targets.
_FORBIDDEN_IN_PARSED = (
    "/var/lib/sherloc",
    "/etc/sherloc",
    "m2020-phase",
)

_SECRET_SHAPED_PATTERN = re.compile(
    r"[A-Z][A-Z0-9_]*_(TOKEN|SECRET|KEY|PASSWORD)\s*=\s*[A-Za-z0-9+/_-]{8,}"
)


def _parsed():
    return yaml.safe_load(COMPOSE_EXAMPLE.read_text())


def test_compose_example_exists():
    assert COMPOSE_EXAMPLE.is_file()


def test_compose_example_parses_as_yaml():
    data = _parsed()
    assert isinstance(data, dict)
    assert "services" in data and data["services"], "compose must declare services"


def test_every_service_pins_archaeon_ai_image():
    data = _parsed()
    for name, svc in data["services"].items():
        image = svc.get("image", "")
        assert image.startswith("ghcr.io/archaeon-ai/sherloc-pipeline:v"), (
            f"service {name!r} image must start with "
            f"ghcr.io/archaeon-ai/sherloc-pipeline:v, got {image!r}"
        )


def test_every_service_has_healthcheck_to_api_health():
    data = _parsed()
    for name, svc in data["services"].items():
        hc = svc.get("healthcheck", {})
        test = hc.get("test", [])
        joined = " ".join(test) if isinstance(test, list) else str(test)
        assert "http://localhost:8000/api/health" in joined, (
            f"service {name!r} healthcheck must hit /api/health on 8000"
        )


def test_every_service_mounts_data_volume():
    data = _parsed()
    for name, svc in data["services"].items():
        volumes = svc.get("volumes", [])
        # Both short ("host:/data") and long-syntax dict forms count.
        mounts = []
        for v in volumes:
            if isinstance(v, str):
                # Short form: "src:dst[:opts]"; target is the second field.
                parts = v.split(":")
                if len(parts) >= 2:
                    mounts.append(parts[1])
            elif isinstance(v, dict):
                mounts.append(v.get("target", ""))
        assert "/data" in mounts, (
            f"service {name!r} must mount a volume to /data, got {volumes!r}"
        )


def test_parsed_config_has_no_operator_local_strings():
    """Comments referring to 'downstream platforms' generically are fine,
    but the *parsed* config (image refs, env_file paths, volume mounts)
    must not bake in operator-local paths or platform names."""
    rendered = yaml.safe_dump(_parsed())
    for forbidden in _FORBIDDEN_IN_PARSED:
        assert forbidden not in rendered, (
            f"forbidden string {forbidden!r} appears in parsed compose; "
            "use a placeholder path/name instead"
        )


def test_no_secret_shaped_values_in_file_text():
    """Belt-and-suspenders on top of scripts/check-forbidden-strings.sh:
    no `*_TOKEN=abc123...` style assignments in the example."""
    text = COMPOSE_EXAMPLE.read_text()
    matches = _SECRET_SHAPED_PATTERN.findall(text)
    assert not matches, f"secret-shaped tokens leaked into compose example: {matches}"
