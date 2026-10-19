"""Offline reference release commands; uploads and activation stay separate."""

from dataclasses import asdict
from pathlib import Path

import typer

from sherloc_pipeline.services.reference_import import import_release

app = typer.Typer(help="Import immutable reference releases from explicit local files.")


@app.command("import")
def import_command(
    snapshot: Path = typer.Option(..., "--snapshot", exists=True, file_okay=False),
    database: Path = typer.Option(..., "--database", dir_okay=False),
    objects: Path = typer.Option(..., "--objects", file_okay=False),
    serving_policy: Path | None = typer.Option(None, "--serving-policy", exists=True, dir_okay=False),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Validate snapshot.json, native NPZ arrays and source bytes without network access."""
    import json

    try:
        report = import_release(snapshot, database, objects, serving_policy, dry_run)
    except (ValueError, OSError, KeyError) as exc:
        typer.echo(f"Reference import failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(asdict(report), sort_keys=True))
