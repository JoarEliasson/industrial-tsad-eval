from __future__ import annotations

from pathlib import Path

from industrial_tsad_eval.application.validation import ValidatePreparedDataset


def test_prepared_dataset_validation_accepts_opcua_fixture(opcua_prepared: Path):
    report = ValidatePreparedDataset(opcua_prepared).run()

    assert report.ok
    assert report.subject == "OPCUA_SYNTH"


def test_prepared_dataset_validation_reports_missing_contract_files(tmp_path: Path):
    broken = tmp_path / "broken"
    (broken / "runs").mkdir(parents=True)

    report = ValidatePreparedDataset(broken).run()

    assert not report.ok
    assert "Missing required file: meta/manifest.json" in report.errors
