from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from typer.testing import CliRunner

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.evidence import GenerateEvidence
from industrial_tsad_eval.application.rq3 import RunReplaySuite
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.domain.llm import LLMMessage, LLMProviderConfig, LLMRequest
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_reproduction_config,
    load_rq3_config,
    write_default_reproduction_config,
)
from industrial_tsad_eval.interfaces.cli.main import app
from industrial_tsad_eval.plugins.providers import default_llm_provider_registry
from industrial_tsad_eval.plugins.registry import default_detector_registry

runner = CliRunner()


def test_provider_registry_contains_reproducibility_and_cloud_shapes():
    registry = default_llm_provider_registry()

    assert registry.names() == [
        "anthropic",
        "fake",
        "google",
        "llama-cpp",
        "openai",
        "openai-compatible",
        "xai",
    ]
    llama = registry.get("llama-cpp").default_config()
    assert llama.base_url == "http://127.0.0.1:8080/v1"
    assert registry.get("fake").create(registry.get("fake").default_config()).healthcheck().ok


def test_openai_compatible_provider_protocol_with_local_stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleStubHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        registry = default_llm_provider_registry()
        provider = registry.get("llama-cpp").create(
            LLMProviderConfig(
                name="llama-cpp",
                model="stub-model",
                base_url=base_url,
                timeout_s=5.0,
            )
        )

        health = provider.healthcheck()
        response = provider.generate(
            LLMRequest(
                messages=[
                    LLMMessage(role="system", content="Use cited evidence."),
                    LLMMessage(role="user", content="Summarize the event [C1]."),
                ]
            )
        )

        assert health.ok
        assert response.provider == "llama-cpp"
        assert "[C1]" in response.text
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def test_rq3_config_parsing_resolves_paths(tmp_path: Path, opcua_prepared: Path):
    config_path = _write_rq3_config(tmp_path / "rq3.toml", opcua_prepared)

    config = load_rq3_config(config_path)

    assert config.prepared == str(opcua_prepared.resolve())
    assert config.provider.name == "fake"
    assert config.cases_per_dataset == 2


def test_rq3_replay_suite_on_opcua_evidence(tmp_path: Path, opcua_prepared: Path):
    scores, eval_dir, evidence = _score_eval_evidence(tmp_path, opcua_prepared)
    config = load_rq3_config(_write_rq3_config(tmp_path / "rq3.toml", opcua_prepared))

    result = RunReplaySuite(
        config=config,
        evidence=evidence,
        out=tmp_path / "rq3",
        provider_registry=default_llm_provider_registry(),
        benchmark=eval_dir,
    ).run()

    assert scores.exists()
    assert result.ok
    assert (tmp_path / "rq3" / "rq3_summary.json").exists()
    assert result.metrics.runs_evaluated >= 1
    assert "citation_compliance_proxy" in result.metrics.to_dict()


def test_reproduction_config_and_cli_smoke_run(tmp_path: Path, opcua_prepared: Path):
    config_path = tmp_path / "reproduction.toml"
    write_default_reproduction_config(config_path, profile="thesis-smoke")
    _replace_config_path(config_path, "examples/generated/OPCUA_SYNTH", opcua_prepared)

    loaded = load_reproduction_config(config_path)
    assert loaded.name == "thesis-smoke"

    plan = runner.invoke(app, ["reproduce", "plan", "--config", str(config_path)])
    assert plan.exit_code == 0, plan.output

    run = runner.invoke(
        app,
        [
            "reproduce",
            "run",
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "repro"),
            "--run-id",
            "smoke",
        ],
    )
    assert run.exit_code == 0, run.output
    run_root = tmp_path / "repro" / "smoke"
    assert (run_root / "benchmark" / "summary.json").exists()
    assert (run_root / "rq3" / "rq3_summary.json").exists()
    assert (run_root / "summaries" / "thesis_crosswalk.md").exists()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True


def test_rq3_cli_commands(tmp_path: Path, opcua_prepared: Path):
    _, eval_dir, evidence = _score_eval_evidence(tmp_path, opcua_prepared)
    config_path = _write_rq3_config(tmp_path / "rq3_cli.toml", opcua_prepared)

    providers = runner.invoke(app, ["rq3", "providers"])
    assert providers.exit_code == 0, providers.output
    assert "llama-cpp" in providers.output

    preflight = runner.invoke(app, ["rq3", "preflight", "--config", str(config_path)])
    assert preflight.exit_code == 0, preflight.output

    run = runner.invoke(
        app,
        [
            "rq3",
            "run",
            "--config",
            str(config_path),
            "--benchmark",
            str(eval_dir),
            "--evidence",
            str(evidence),
            "--out",
            str(tmp_path / "rq3-cli"),
        ],
    )
    assert run.exit_code == 0, run.output
    summarize = runner.invoke(app, ["rq3", "summarize", "--run", str(tmp_path / "rq3-cli")])
    assert summarize.exit_code == 0, summarize.output


def _score_eval_evidence(tmp_path: Path, prepared: Path) -> tuple[Path, Path, Path]:
    scores = tmp_path / "scores"
    eval_dir = tmp_path / "eval"
    evidence = tmp_path / "evidence"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()
    EvaluateScores(prepared=prepared, scores=scores, out=eval_dir).run()
    GenerateEvidence(prepared=prepared, scores=scores, eval_dir=eval_dir, out=evidence).run()
    return scores, eval_dir, evidence


def _write_rq3_config(path: Path, prepared: Path) -> Path:
    escaped = str(prepared.resolve()).replace("\\", "\\\\")
    path.write_text(
        f"""[rq3]
suite_id = "rq3-test"
prepared = "{escaped}"
cases_per_dataset = 2
top_k = 6
minimum_supported_claims = 1
prompt_budget_chars = 8000
query_template = "For {{dataset}} {{event_id}}: cite causes and checks for {{top_variables}}."

[rq3.provider]
name = "fake"
model = "fake-rq3"
""",
        encoding="utf-8",
    )
    return path


def _replace_config_path(path: Path, placeholder: str, prepared: Path) -> None:
    escaped = str(prepared.resolve()).replace("\\", "\\\\")
    path.write_text(
        path.read_text(encoding="utf-8").replace(placeholder, escaped),
        encoding="utf-8",
    )


class _OpenAICompatibleStubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json_response({"object": "list", "data": [{"id": "stub-model"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self._json_response(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "- Stubbed provider path is healthy [C1].",
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json_response(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
