"""GET /api/config and GET /api/config/access-mode endpoints."""

import hashlib
import json
import os

from fastapi import APIRouter, Request

from sherloc_pipeline.web.schemas import (
    API_SCHEMA_VERSION,
    AccessModeResponse,
    AuthConfig,
    ConfigResponse,
    FeaturesConfig,
)


# Sentinel for an explicit feature-disabled env value. Anything else
# (absent, "enabled", or a misspelling) leaves the feature on so a
# typo in deploy templates can't silently hide a feature — only the
# literal "disabled" (case-insensitive) opts a feature out.
_FEATURE_DISABLED = "disabled"


def feature_pds_browser_enabled() -> bool:
    """Resolve the SHERLOC_FEATURE_PDS_BROWSER env var.

    Production env templates set this to ``disabled`` to hide the PDS
    Browser tab + 404 its API endpoints (issue #21). Local dev +
    legacy deployments leave it unset → tab visible. Re-export so route guards
    elsewhere (``routes/pds.py``) consult the same logic.
    """
    return os.environ.get("SHERLOC_FEATURE_PDS_BROWSER", "").strip().lower() != _FEATURE_DISABLED

router = APIRouter(prefix="/api", tags=["config"])


def _build_auth_config() -> AuthConfig:
    """Read SHERLOC_AUTH_* env vars into the SPA-facing auth metadata.

    Per §13.4: the same SPA build runs against any auth backend without
    rebuild. The SPA reads this block at startup to decide whether to
    bootstrap ``@auth0/auth0-spa-js``, treat the page as a CF Access
    cookie session, or skip auth entirely (dev mode).
    """
    mode = os.environ.get("SHERLOC_AUTH_MODE", "cf-access")
    if mode not in ("auth0", "cf-access", "dev"):
        # Validation lives in config_check; surface a safe default here
        # so the endpoint never raises during a misconfigured boot.
        mode = "cf-access"

    if mode != "auth0":
        return AuthConfig(auth_mode=mode)

    # Note: ``role_claim_uri`` is left at its schema default (None) since
    # the §2.6.1 contract supersedes the legacy ``{role_claim_uri}/roles``
    # path — the SPA does not consume this field anymore. The schema
    # field is preserved for SPA back-compat across the v4.0 → v4.1
    # cutover and will be retired in a follow-up alongside the SPA
    # types cleanup.
    return AuthConfig(
        auth_mode="auth0",
        auth0_domain=os.environ.get("SHERLOC_AUTH0_DOMAIN"),
        auth0_client_id=os.environ.get("SHERLOC_AUTH0_SPA_CLIENT_ID"),
        auth0_audience=os.environ.get("SHERLOC_AUTH0_AUDIENCE"),
    )


@router.get("/config", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    """Return the currently active fitting and preprocessing configuration.

    Also returns an ``auth`` block (§13.4) so the SPA can bootstrap the
    correct auth backend at runtime without a rebuild.
    """
    config = request.app.state.config

    fitting_json = json.dumps(config.fitting, sort_keys=True, default=str)
    config_hash = f"sha256:{hashlib.sha256(fitting_json.encode()).hexdigest()[:12]}"

    cal = config.wavelength
    calibration = {
        "version": "loupe_v5.1.5a",
        "laser_wavelength_nm": cal.laser_wavelength,
        "cutoff_channel": cal.cutoff_channel,
        "n_channels": cal.n_channels,
        "raman_coefficients": cal.raman_coefficients,
        "fluorescence_coefficients": cal.fluorescence_coefficients,
    }

    return ConfigResponse(
        config_hash=config_hash,
        fitting=config.fitting,
        fluorescence_fitting=config.fluorescence_fitting,
        preprocessing=config.preprocessing,
        calibration=calibration,
        auth=_build_auth_config(),
        features=FeaturesConfig(pds_browser=feature_pds_browser_enabled()),
    )


@router.get("/config/access-mode", response_model=AccessModeResponse)
def get_access_mode(request: Request) -> AccessModeResponse:
    """Return the current data access mode (internal or public)."""
    access_mode = getattr(request.app.state, "access_mode", "internal")
    return AccessModeResponse(access_mode=access_mode)
