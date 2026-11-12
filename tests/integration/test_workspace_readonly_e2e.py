"""run_scan must leave the input Loupe workspace byte-identical (issue #131).

Runs on a copy of the committed sol_0921 fixture so a regression cannot
modify the fixture tree itself.
"""

import hashlib
import shutil

from sherloc_pipeline.services.preprocessing import PreprocessingService

SOL_DIR = "loupe/sol_0921"
WORKSPACE_REL = "detail_1/SrlcSpecSpecSohRaw_0748731411-51550-1_Loupe_working"


def snapshot(directory):
    return {
        p.relative_to(directory): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def test_run_scan_does_not_modify_the_workspace(tmp_path, fixtures_path):
    data_dir = tmp_path / "loupe"
    shutil.copytree(fixtures_path / SOL_DIR, data_dir / "sol_0921")
    # Drop the workspace's own normalized file so the run has to produce one.
    workspace = data_dir / "sol_0921" / WORKSPACE_REL
    (workspace / "darkSubSpectraN.csv").unlink()
    before = snapshot(data_dir)
    results_dir = tmp_path / "results"

    PreprocessingService().run_scan(
        sol="0921",
        target="Readonly_E2E",
        scan="detail_1",
        data_dir=data_dir,
        results_dir=results_dir,
        generate_plots=False,
        despike_method="none",
        baseline_r1=False,
    )

    assert snapshot(data_dir) == before
    assert (results_dir / "Readonly_E2E" / "0921_detail_1" / "darkSubSpectraN.csv").is_file()
