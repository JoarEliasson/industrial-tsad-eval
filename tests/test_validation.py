from __future__ import annotations

from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture


def test_prepared_fixture_validates(tmp_path):
    prepared = make_opcua_fixture(tmp_path / "examples")

    report = ValidatePreparedDataset(prepared).run()

    assert report.ok
    assert report.errors == []


def test_prepared_validation_reports_missing_contract_files(tmp_path):
    prepared = tmp_path / "BROKEN"
    (prepared / "runs").mkdir(parents=True)

    report = ValidatePreparedDataset(prepared).run()

    assert not report.ok
    assert "Missing required file: meta/manifest.json" in report.errors
