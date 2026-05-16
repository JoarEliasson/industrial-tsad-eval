from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from industrial_tsad_eval.application.validation import (
    ValidatePreparedDataset,
    ValidatePreparedDatasetCached,
)


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


def test_prepared_dataset_validation_reads_only_timestamp_column(
    monkeypatch,
    opcua_prepared: Path,
):
    calls: list[object] = []
    original_read_parquet = pd.read_parquet

    def tracking_read_parquet(*args, **kwargs):
        calls.append(kwargs.get("columns"))
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracking_read_parquet)

    report = ValidatePreparedDataset(opcua_prepared).run()

    assert report.ok
    assert calls
    assert all(columns == ["ts_ns"] for columns in calls)


def test_prepared_validation_cache_hits_when_fingerprint_is_unchanged(
    monkeypatch,
    opcua_prepared: Path,
    tmp_path: Path,
):
    cache_root = tmp_path / "cache"

    first_report, first_cache = ValidatePreparedDatasetCached(opcua_prepared, cache_root).run()
    assert first_report.ok
    assert first_cache["validation_cache"] == "miss"

    def fail_full_validation(_self):
        raise AssertionError("full validation should not run on cache hit")

    monkeypatch.setattr(ValidatePreparedDataset, "run", fail_full_validation)
    second_report, second_cache = ValidatePreparedDatasetCached(opcua_prepared, cache_root).run()

    assert second_report.ok
    assert second_cache["validation_cache"] == "hit"


def test_prepared_dataset_validation_reports_non_numeric_columns(tmp_path: Path):
    prepared = tmp_path / "prepared"
    run_dir = prepared / "runs" / "dataset" / "train" / "run_001"
    run_dir.mkdir(parents=True)
    (prepared / "meta").mkdir(parents=True)
    (prepared / "events").mkdir(parents=True)

    (prepared / "meta" / "manifest.json").write_text(
        json.dumps({"dataset": "BROKEN", "runs": {"run_ids": ["dataset/train/run_001"]}}),
        encoding="utf-8",
    )
    (prepared / "meta" / "schema.json").write_text(
        json.dumps(
            {
                "tags": [
                    {"browse_path": "Plant/Broken/string_feature"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (prepared / "meta" / "splits.json").write_text("{}", encoding="utf-8")
    (prepared / "events" / "events.jsonl").write_text("", encoding="utf-8")
    pd.DataFrame(
        {
            "ts_ns": [1, 2, 3],
            "Plant/Broken/string_feature": ["a", "b", "c"],
        }
    ).to_parquet(run_dir / "timeseries.parquet", index=False)
    (run_dir / "run_meta.json").write_text("{}", encoding="utf-8")

    report = ValidatePreparedDataset(prepared).run()

    assert not report.ok
    assert any("non-numeric dtype" in error for error in report.errors)
