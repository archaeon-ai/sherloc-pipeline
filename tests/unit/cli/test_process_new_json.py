"""Every process-new execution path reserves stdout for one JSON envelope."""

import importlib
import json
import zipfile
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from sherloc_pipeline.database.connection import get_engine, get_session
from sherloc_pipeline.database.models import Base, ScanORM, SolORM
from sherloc_pipeline.services.base import ServiceResult

cli = importlib.import_module('sherloc_pipeline.cli.app')


@pytest.fixture
def invoke(tmp_path):
    def run(path, *extra):
        return CliRunner().invoke(cli.app, [
            '--json', 'process-new', str(path), '--skip-ingest',
            '--database', str(tmp_path / 'test.db'),
            '--data-dir', str(tmp_path / 'data'),
            '--results-dir', str(tmp_path / 'results'), *extra,
        ])
    return run


@pytest.mark.parametrize('case', ['missing', 'extraction', 'file', 'unknown-sol', 'bad-zip'])
def test_validation_failure_has_one_stdout_error(tmp_path, invoke, case):
    path = tmp_path / 'sol_0001'
    if case == 'extraction':
        path = path.with_suffix('.zip')
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('other/input.txt', 'synthetic')
    elif case == 'file':
        path.write_text('synthetic')
    elif case == 'unknown-sol':
        path = tmp_path / 'unknown'
        path.mkdir()
    elif case == 'bad-zip':
        path = path.with_suffix('.zip')
        path.write_text('synthetic invalid archive')
    result = invoke(path)
    assert result.exit_code == 1, result.output
    error = json.loads(result.stdout)
    assert error['exit_code'] == 1
    assert error['error_type']
    assert error['message']


def seed_scans(tmp_path, rows):
    engine = get_engine(tmp_path / 'test.db')
    Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add(SolORM(sol_number=1, data_source='loupe'))
        session.flush()
        for index, (target, target_type) in enumerate(rows):
            session.add(ScanORM(
                id=f'synthetic-{index}', sol_number=1, scan_name=f'scan-{index}',
                target=target, target_type=target_type, scan_id=f'synthetic-{index}',
                sclk_start=index, n_points=1,
            ))
    path = tmp_path / 'sol_0001'
    path.mkdir()
    return path


@pytest.mark.parametrize('rows,status', [
    ([], 'empty-ingest'),
    ([(None, None)], 'no-target-scans'),
    ([('Synthetic calibration', 'cal_target')], 'no-science-scans'),
])
def test_no_fittable_scans_has_counts_and_reason(tmp_path, invoke, rows, status):
    result = invoke(seed_scans(tmp_path, rows))
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert output['command'] == 'process-new'
    assert output['result']['sol'] == 1
    assert output['result']['scans_processed'] == 0
    assert output['result']['errors'] == []
    assert output['metadata']['processing_status'] == status
    assert output['metadata']['scans_available'] == len(rows)
    assert output['metadata']['targetless_scans'] == sum(not target for target, _ in rows)
    assert 'No fittable science scans' in result.stderr


@pytest.mark.parametrize('fail', [False, True])
def test_pipeline_output_is_single_json(tmp_path, invoke, monkeypatch, fail):
    path = seed_scans(tmp_path, [('Synthetic rock', 'mars_target')])
    pipeline = Mock()
    def fit(**kwargs):
        pipeline.call_args.kwargs['console'].print('synthetic fitting progress')
        if fail:
            raise ValueError('synthetic fit failure')
        return ServiceResult(summary='synthetic fitted')
    pipeline.return_value.run_full_pipeline.side_effect = fit
    monkeypatch.setattr(cli, 'PipelineService', pipeline)
    result = invoke(path)
    assert result.exit_code == int(fail), result.output
    output = json.loads(result.stdout)
    assert output['result']['scans_processed'] == 1
    assert bool(output['result']['errors']) is fail
    assert 'synthetic fitting progress' in result.stderr


@pytest.mark.parametrize('zip_input', [False, True])
def test_dry_run_has_single_json_and_no_fits(tmp_path, invoke, monkeypatch, zip_input):
    path = seed_scans(tmp_path, [('Synthetic rock', 'mars_target')])
    if zip_input:
        path = tmp_path / 'sol_0002.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('sol_0002/scan_Loupe_working/input.txt', 'synthetic')
    pipeline = Mock()
    monkeypatch.setattr(cli, 'PipelineService', pipeline)
    result = invoke(path, '--dry-run')
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert output['metadata']['dry_run'] is True
    assert output['result']['scans_processed'] == 0
    pipeline.assert_not_called()


@pytest.mark.parametrize('option,value', [('--trim-pct', '51'), ('--model-selection', 'invalid')])
def test_invalid_override_has_one_error(tmp_path, invoke, option, value):
    path = tmp_path / 'sol_0001'
    path.mkdir()
    result = invoke(path, option, value)
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)['error_type'] == 'ValueError'
    assert option in result.stderr


def test_override_messages_stay_off_stdout(tmp_path, invoke, monkeypatch):
    import copy
    config = cli.get_config()
    monkeypatch.setattr(config, 'preprocessing', copy.deepcopy(config.preprocessing))
    monkeypatch.setattr(config, 'fitting', copy.deepcopy(config.fitting))
    path = seed_scans(tmp_path, [('Synthetic calibration', 'cal_target')])
    result = invoke(path, '--trim-pct', '5', '--model-selection', 'ftest', '--despike-method', 'none')
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['result']['scans_processed'] == 0
    assert 'Trim override' in result.stderr
    assert 'Model selection override' in result.stderr
    assert 'Despike method override' in result.stderr
