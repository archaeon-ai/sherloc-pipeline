"""Startup access/storage pairing with entirely invented environment values."""

import os
from unittest.mock import MagicMock

import pytest

from sherloc_pipeline.web import config_check
from sherloc_pipeline.web.app import create_app


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate the environment without loading config files or database fixtures."""
    for key in list(os.environ):
        if key.startswith(("SHERLOC_", "AWS_")) or key in ("PHASE_TIER", "PHASE_DATABASE_PATH"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHERLOC_DB", ":memory:")
    monkeypatch.setenv("SHERLOC_AUTH0_DOMAIN", "invented.invalid")
    monkeypatch.setenv("SHERLOC_AUTH0_AUDIENCE", "https://invented.invalid/api")
    monkeypatch.setenv("SHERLOC_AUTH0_SPA_CLIENT_ID", "invented-spa")
    monkeypatch.setenv("SHERLOC_AUTH0_IDENTITY_CLAIM_URI", "https://invented.invalid/identity")
    monkeypatch.setenv("SHERLOC_ALEMBIC_TARGET", config_check.RETAINED_PATH_REVISION)
    monkeypatch.setenv("SHERLOC_CF_TEAM_DOMAIN", "invented.cloudflareaccess.com")
    monkeypatch.setenv("SHERLOC_CF_AUDIENCE", "invented-audience")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "invented-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "invented-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.invalid")
    return monkeypatch


def _message(access_mode, tier):
    expected = "public" if access_mode == "public" else "team"
    return (
        f"SHERLOC_ACCESS_MODE={access_mode!r} requires "
        f"PHASE_TIER={expected!r}; got PHASE_TIER={tier!r}"
    )


@pytest.mark.parametrize("access_mode,tier", [("internal", "team"), ("public", "public")])
def test_auth0_matching_pairs_pass(clean_env, access_mode, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    clean_env.setenv("PHASE_TIER", tier)
    assert config_check.validate() == []


@pytest.mark.parametrize("auth_mode", ["auth0", "cf-access", "dev"])
@pytest.mark.parametrize("access_mode,tier", [("public", "team"), ("internal", "public")])
def test_mismatched_pairs_fail(clean_env, auth_mode, access_mode, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", auth_mode)
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    clean_env.setenv("PHASE_TIER", tier)
    assert config_check.validate() == [_message(access_mode, tier)]


@pytest.mark.parametrize("tier", ["team", "public"])
def test_unset_access_mode_defaults_to_internal(clean_env, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("PHASE_TIER", tier)
    expected = [] if tier == "team" else [_message("internal", tier)]
    assert config_check.validate() == expected


@pytest.mark.parametrize("auth_mode", ["auth0", "cf-access", "dev"])
@pytest.mark.parametrize("access_mode", ["internal", "public"])
def test_unset_tier_preserves_existing_requirements(clean_env, auth_mode, access_mode):
    clean_env.setenv("SHERLOC_AUTH_MODE", auth_mode)
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    expected = ["missing required variable: PHASE_TIER"] if auth_mode == "auth0" else []
    assert config_check.validate() == expected


@pytest.mark.parametrize("access_mode,tier", [("internal", " TEAM "), ("public", " PUBLIC ")])
def test_pairing_uses_storage_tier_normalization(clean_env, access_mode, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    clean_env.setenv("PHASE_TIER", tier)
    assert config_check.validate() == []


@pytest.mark.parametrize("tier", ["", "  "])
def test_blank_tier_counts_as_unset_outside_auth0(clean_env, tier):
    # A blank PHASE_TIER in a dev env keeps booting as before; auth0 mode
    # already rejects it through the R2 check, so nothing is lost.
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("PHASE_TIER", tier)
    assert config_check.validate() == []
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    assert config_check.validate() == ["missing required variable: PHASE_TIER"]


@pytest.mark.parametrize("access_mode,tier", [("public", "team"), ("internal", "public")])
def test_direct_app_startup_rejects_mismatches(clean_env, access_mode, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    clean_env.setenv("PHASE_TIER", tier)
    # Engine/config stubs avoid all files and data, including on the baseline.
    engine, config = MagicMock(), MagicMock()
    with pytest.raises(ValueError) as exc:
        create_app(engine=engine, config=config)
    assert str(exc.value) == _message(access_mode, tier)


def test_direct_app_startup_checks_effective_explicit_mode(clean_env):
    clean_env.setenv("SHERLOC_ACCESS_MODE", "internal")
    clean_env.setenv("PHASE_TIER", "team")
    with pytest.raises(ValueError) as exc:
        create_app(engine=MagicMock(), config=MagicMock(), access_mode="public")
    assert str(exc.value) == _message("public", "team")


@pytest.mark.parametrize("access_mode,tier", [("internal", "team"), ("public", "public")])
def test_direct_app_startup_matching_pairs_pass(clean_env, access_mode, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", "auth0")
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    clean_env.setenv("PHASE_TIER", tier)
    app = create_app(engine=MagicMock(), config=MagicMock())
    assert app.state.access_mode == access_mode
    app.state.job_queue.shutdown()
    app.state.map_executor.shutdown(wait=False)


@pytest.mark.parametrize("access_mode", ["internal", "public"])
def test_direct_app_startup_unset_tier_still_passes(clean_env, access_mode):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    app = create_app(engine=MagicMock(), config=MagicMock())
    assert app.state.access_mode == access_mode
    app.state.job_queue.shutdown()
    app.state.map_executor.shutdown(wait=False)


def _invalid_mode_message(access_mode):
    return (
        f"access mode has unrecognized value: {access_mode!r} "
        f"(expected one of ['internal', 'public'])"
    )


@pytest.mark.parametrize("tier", [None, "team", "public"])
@pytest.mark.parametrize("access_mode", ["pubilc", "PUBLIC", ""])
def test_direct_app_startup_rejects_invalid_env_mode(clean_env, access_mode, tier):
    # A typo must not boot unrestricted; the entrypoint rejects it too.
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    clean_env.setenv("SHERLOC_ACCESS_MODE", access_mode)
    if tier is not None:
        clean_env.setenv("PHASE_TIER", tier)
    with pytest.raises(ValueError) as exc:
        create_app(engine=MagicMock(), config=MagicMock())
    assert str(exc.value) == _invalid_mode_message(access_mode)
    assert (
        f"SHERLOC_ACCESS_MODE has unrecognized value: {access_mode!r} "
        f"(expected one of ['internal', 'public'])"
    ) in config_check.validate()


@pytest.mark.parametrize("tier", [None, "team", "public"])
@pytest.mark.parametrize("access_mode", ["pubilc", "Public"])
def test_direct_app_startup_rejects_invalid_explicit_mode(clean_env, access_mode, tier):
    clean_env.setenv("SHERLOC_AUTH_MODE", "dev")
    if tier is not None:
        clean_env.setenv("PHASE_TIER", tier)
    with pytest.raises(ValueError) as exc:
        create_app(engine=MagicMock(), config=MagicMock(), access_mode=access_mode)
    assert str(exc.value) == _invalid_mode_message(access_mode)
