from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Generator[Path]:
    base = PROJECT_ROOT / ".test-output" / request.node.name.replace("\\", "_").replace("/", "_")
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    yield path


@pytest.fixture
def opcua_prepared(tmp_path: Path) -> Path:
    from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture

    return make_opcua_fixture(tmp_path / "examples")


def write_swat_raw(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    normal = _industrial_frame(220)
    normal["Normal/Attack"] = "Normal"
    attack = _industrial_frame(220, anomaly=True)
    attack["Normal/Attack"] = [
        "Attack" if 140 <= index <= 170 else "Normal" for index in range(220)
    ]
    normal.to_csv(root / "SWaT_Dataset_Normal.csv", index=False)
    attack.to_csv(root / "SWaT_Dataset_Attack.csv", index=False)
    return root


def write_hai_raw(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train = _industrial_frame(220).rename(columns={"SensorA": "P1_FIT101", "ValveA": "P2_MV201"})
    test = _industrial_frame(220, anomaly=True).rename(
        columns={"SensorA": "P1_FIT101", "ValveA": "P2_MV201"}
    )
    test["attack"] = [1 if 130 <= index <= 155 else 0 for index in range(220)]
    train.to_csv(root / "hai_train.csv", index=False)
    test.to_csv(root / "hai_test_attack.csv", index=False)
    return root


def write_tep_raw(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    train = pd.DataFrame(
        {
            "simulationRun": 1,
            "sample": np.arange(1, 221),
            "faultNumber": 0,
            "xmeas_1": 10 + np.sin(np.arange(220) / 12),
            "xmv_1": 20 + np.cos(np.arange(220) / 18),
        }
    )
    test = train.copy()
    test["faultNumber"] = 1
    test.loc[160:, "xmeas_1"] += 3
    test.to_csv(root / "d00_train.csv", index=False)
    test.to_csv(root / "d01_test.csv", index=False)
    return root


def write_hai_cpps_raw(root: Path) -> Path:
    normal_dir = root / "normal_scenario"
    attack_dir = root / "anomaly_scenario"
    normal_dir.mkdir(parents=True, exist_ok=True)
    attack_dir.mkdir(parents=True, exist_ok=True)
    _industrial_frame(220).to_csv(normal_dir / "continuous.csv", index=False)
    _industrial_frame(220, anomaly=True).to_csv(attack_dir / "continuous.csv", index=False)
    (attack_dir / "sim_setup.json").write_text(
        '{"attack_start": 120, "attack_duration": 30, "tag": "SensorA"}',
        encoding="utf-8",
    )
    return root


def _industrial_frame(rows: int, anomaly: bool = False) -> pd.DataFrame:
    index = np.arange(rows)
    sensor = 10.0 + np.sin(index / 12.0)
    flow = 5.0 + np.cos(index / 20.0)
    valve = 1.0 + 0.1 * np.sin(index / 18.0)
    if anomaly:
        sensor[120:170] += np.linspace(1.0, 4.0, 50)
        flow[120:170] -= 1.0
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2026-01-01", periods=rows, freq="s"),
            "SensorA": sensor,
            "FlowA": flow,
            "ValveA": valve,
        }
    )
