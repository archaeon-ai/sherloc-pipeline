"""Offline one-point provisional candidate support; activation stays separate."""

import os
from pathlib import Path

import typer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from sherloc_pipeline.models.identification import IdentificationRequest
from sherloc_pipeline.services.identification import IdentificationService, LocalPointResolver, deny_source
from sherloc_pipeline.services.reference_store import LocalReferenceStore
from sherloc_pipeline.web.config_check import VALID_ACCESS_MODES, access_tier_pairing_error
from sherloc_pipeline.web.feature_policy import SourceDenylist, load_feature_policy

app = typer.Typer(help="Offline frozen provisional Raman candidate support (default off).")


@app.command("point")
def identify_point(
    request_json: str = typer.Option(..., "--request-json", help="Pinned one-point request JSON, not a file."),
    database: Path = typer.Option(..., "--database", dir_okay=False),
    objects: Path = typer.Option(..., "--objects", file_okay=False),
    sources: Path = typer.Option(..., "--sources", file_okay=False),
    source_denylist_json: str | None = typer.Option(
        None, "--source-denylist-json", help="Additional source exclusions as inline JSON: sols, product_ids, target_names.",
    ),
):
    """Resolve one local committed point; never fit thresholds or scan all points."""
    try:
        tier = os.environ.get("PHASE_TIER", "").strip().lower()
        if tier != "team":
            raise ValueError("experimental identification requires PHASE_TIER=team")
        access_mode = os.environ.get("SHERLOC_ACCESS_MODE", "internal")
        if access_mode not in VALID_ACCESS_MODES:
            raise ValueError("unrecognized SHERLOC_ACCESS_MODE")
        pairing_error = access_tier_pairing_error(access_mode)
        if pairing_error:
            raise ValueError(pairing_error)
        request = IdentificationRequest.model_validate_json(request_json)
        policy = load_feature_policy(tier)
        if source_denylist_json is not None:
            extra = SourceDenylist.model_validate_json(source_denylist_json)
            configured = policy.identification.source_denylist
            combined = SourceDenylist(**{
                key: sorted(set(getattr(configured, key) + getattr(extra, key)))
                for key in ("sols", "product_ids", "target_names")
            })
            policy = policy.model_copy(update={"identification": policy.identification.model_copy(
                update={"source_denylist": combined},
            )})
        deny_source(request.selection, policy.identification.source_denylist)
        if not policy.identification.enabled or not policy.identification.experimental_candidates:
            raise ValueError("experimental identification is disabled")
        engine = create_engine(f"sqlite:///file:{database.resolve()}?mode=ro&uri=true")
        try:
            with Session(engine) as session:
                result = IdentificationService(session, LocalReferenceStore(objects),
                                               LocalPointResolver(sources, policy.identification.source_denylist),
                                               policy, tier).identify(request)
                typer.echo(result.model_dump_json())
        finally:
            engine.dispose()
    except (ValueError, OSError, KeyError) as exc:
        typer.echo(f"Identification failed: {exc}", err=True)
        raise typer.Exit(1) from exc
