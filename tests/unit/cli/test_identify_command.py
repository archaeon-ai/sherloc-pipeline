"""Command discovery and invocation after retirement of the identify group."""

from typer.testing import CliRunner

from sherloc_pipeline.cli.app import app


def test_root_help_omits_identify():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "identify" not in result.output.split()


def test_identify_point_is_an_unknown_command():
    result = CliRunner().invoke(app, ["identify", "point", "--help"])
    assert result.exit_code == 2
    assert "No such command 'identify'" in result.output
