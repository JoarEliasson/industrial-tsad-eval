from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from typer.testing import CliRunner

from industrial_tsad_eval.application.assistant_replay import RunAssistantReplaySuite
from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.evidence import GenerateEvidence
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.domain.assistant_replay import (
    THESIS_ASSISTANT_QUERY_TEMPLATE,
    AssistantReplayConfig,
    ClaimEvaluation,
    DraftResponse,
)
from industrial_tsad_eval.domain.llm import (
    LLMMessage,
    LLMProviderConfig,
    LLMRequest,
    LLMStructuredRequest,
)
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_assistant_config,
    load_reproduction_config,
    write_default_reproduction_config,
)
from industrial_tsad_eval.interfaces.cli.main import app
from industrial_tsad_eval.plugins.providers import (
    _schema_instruction,
    default_llm_provider_registry,
)
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
    assert llama.model == "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
    assert llama.base_url == "http://127.0.0.1:8080/v1"
    assert llama.extra["structured_output_mode"] == "json_object"
    assert registry.get("fake").create(registry.get("fake").default_config()).healthcheck().ok


def test_openai_compatible_provider_protocol_with_local_stub():
    _OpenAICompatibleStubHandler.seen_response_formats = []
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
        structured = provider.generate_json(
            LLMStructuredRequest(
                messages=[
                    LLMMessage(role="system", content="Return DraftResponse JSON."),
                    LLMMessage(role="user", content="Summarize the event."),
                ],
                schema_name="DraftResponse",
                json_schema=DraftResponse.model_json_schema(),
            )
        )

        assert health.ok
        assert response.provider == "llama-cpp"
        assert "[C1]" in response.text
        assert structured.payload["checks"]
        assert structured.metadata["response_format"] == {"type": "json_object"}
        assert structured.metadata["failed_attempts"] == []
        seen_types = [
            item.get("type") for item in _OpenAICompatibleStubHandler.seen_response_formats
        ]
        assert seen_types == ["json_object"]
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def test_llama_cpp_json_object_succeeds_without_json_schema_attempts():
    _FallbackStructuredStubHandler.seen_response_formats = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FallbackStructuredStubHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        provider = (
            default_llm_provider_registry()
            .get("llama-cpp")
            .create(
                LLMProviderConfig(
                    name="llama-cpp",
                    model="stub-model",
                    base_url=base_url,
                    timeout_s=5.0,
                )
            )
        )

        structured = provider.generate_json(
            LLMStructuredRequest(
                messages=[LLMMessage(role="user", content="Return DraftResponse JSON.")],
                schema_name="DraftResponse",
                json_schema=DraftResponse.model_json_schema(),
            )
        )

        assert DraftResponse.model_validate(structured.payload).checks
        assert structured.metadata["structured_attempt"] == "json_object"
        assert structured.metadata["failed_attempts"] == []
        seen_types = [
            item.get("type") for item in _FallbackStructuredStubHandler.seen_response_formats
        ]
        assert seen_types == ["json_object"]
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def test_generic_openai_compatible_can_still_use_json_schema():
    _OpenAICompatibleStubHandler.seen_response_formats = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleStubHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        provider = (
            default_llm_provider_registry()
            .get("openai-compatible")
            .create(
                LLMProviderConfig(
                    name="openai-compatible",
                    model="stub-model",
                    base_url=base_url,
                    timeout_s=5.0,
                    extra={
                        "structured_output_mode": "json_schema",
                        "structured_output_allow_fallback": False,
                    },
                )
            )
        )

        structured = provider.generate_json(
            LLMStructuredRequest(
                messages=[LLMMessage(role="user", content="Return DraftResponse JSON.")],
                schema_name="DraftResponse",
                json_schema=DraftResponse.model_json_schema(),
            )
        )

        assert DraftResponse.model_validate(structured.payload).checks
        assert structured.metadata["structured_attempt"] == "json_schema:openai_nested"
        assert structured.metadata["response_format"]["type"] == "json_schema"
        seen_types = [
            item.get("type") for item in _OpenAICompatibleStubHandler.seen_response_formats
        ]
        assert seen_types == ["json_schema"]
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def test_compact_assistant_schema_instructions_are_short_and_specific():
    draft_schema = DraftResponse.model_json_schema()
    referee_schema = ClaimEvaluation.model_json_schema()

    draft_instruction = _schema_instruction("DraftResponse", draft_schema)
    referee_instruction = _schema_instruction("ClaimEvaluation", referee_schema)

    assert len(draft_instruction) < len(json.dumps(draft_schema, sort_keys=True))
    assert len(referee_instruction) < len(json.dumps(referee_schema, sort_keys=True))
    for key in (
        "symptom_summary",
        "likely_causes",
        "checks",
        "recommended_actions",
        "escalation_criteria",
    ):
        assert key in draft_instruction
    for key in (
        "is_supported",
        "entailment_label",
        "entailment_reasoning",
        "final_disposition",
        "rewritten_statement",
    ):
        assert key in referee_instruction


def test_invalid_structured_payload_still_fails_pydantic_validation():
    invalid = {"symptom_summary": "ok", "checks": "not-a-list"}

    try:
        DraftResponse.model_validate(invalid)
    except ValueError:
        pass
    else:  # pragma: no cover - this documents the expected pydantic boundary.
        raise AssertionError("Invalid DraftResponse payload unexpectedly validated.")


def test_failed_assistant_case_writes_diagnostic_artifacts(tmp_path: Path, opcua_prepared: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _InvalidStructuredStubHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        _, eval_dir, evidence = _score_eval_evidence(tmp_path, opcua_prepared)
        config = load_assistant_config(
            _write_assistant_config(tmp_path / "assistant.toml", opcua_prepared)
        )
        config = AssistantReplayConfig(
            suite_id=config.suite_id,
            prepared=config.prepared,
            provider=LLMProviderConfig(
                name="llama-cpp",
                model="stub-model",
                base_url=base_url,
                timeout_s=5.0,
            ),
            query_template=config.query_template,
            cases_per_dataset=1,
            top_k=config.top_k,
            minimum_supported_claims=0,
            prompt_budget_chars=config.prompt_budget_chars,
        )

        result = RunAssistantReplaySuite(
            config=config,
            evidence=evidence,
            out=tmp_path / "assistant-failed",
            provider_registry=default_llm_provider_registry(),
            benchmark=eval_dir,
        ).run()

        run_logs = list((tmp_path / "assistant-failed" / "runs").glob("*/run_log.json"))
        assert result.ok is False
        assert run_logs
        run_log = json.loads(run_logs[0].read_text(encoding="utf-8"))
        provider_response = json.loads(
            (run_logs[0].parent / "provider_response.json").read_text(encoding="utf-8")
        )
        assert run_log["status"] == "failed"
        assert "invalid JSON" in provider_response["error"]
    finally:
        server.shutdown()
        thread.join(timeout=5.0)


def test_fake_provider_is_schema_aware():
    registry = default_llm_provider_registry()
    provider = registry.get("fake").create(registry.get("fake").default_config())

    draft = provider.generate_json(
        LLMStructuredRequest(
            messages=[LLMMessage(role="user", content="planner")],
            schema_name="DraftResponse",
            json_schema=DraftResponse.model_json_schema(),
        )
    )
    referee = provider.generate_json(
        LLMStructuredRequest(
            messages=[LLMMessage(role="user", content="referee")],
            schema_name="ClaimEvaluation",
            json_schema=ClaimEvaluation.model_json_schema(),
        )
    )

    assert DraftResponse.model_validate(draft.payload).checks
    assert ClaimEvaluation.model_validate(referee.payload).final_disposition == "keep"


def test_assistant_config_parsing_resolves_paths(tmp_path: Path, opcua_prepared: Path):
    config_path = _write_assistant_config(tmp_path / "assistant.toml", opcua_prepared)

    config = load_assistant_config(config_path)

    assert config.prepared == str(opcua_prepared.resolve())
    assert config.provider.name == "fake"
    assert config.cases_per_dataset == 2


def test_assistant_replay_suite_on_opcua_evidence(tmp_path: Path, opcua_prepared: Path):
    scores, eval_dir, evidence = _score_eval_evidence(tmp_path, opcua_prepared)
    config = load_assistant_config(
        _write_assistant_config(tmp_path / "assistant.toml", opcua_prepared)
    )

    result = RunAssistantReplaySuite(
        config=config,
        evidence=evidence,
        out=tmp_path / "assistant",
        provider_registry=default_llm_provider_registry(),
        benchmark=eval_dir,
    ).run()

    assert scores.exists()
    assert result.ok
    assert (tmp_path / "assistant" / "assistant_summary.json").exists()
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
    assert (run_root / "assistant" / "assistant_summary.json").exists()
    assert (run_root / "summaries" / "thesis_crosswalk.md").exists()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True


def test_assistant_cli_commands(tmp_path: Path, opcua_prepared: Path):
    _, eval_dir, evidence = _score_eval_evidence(tmp_path, opcua_prepared)
    config_path = _write_assistant_config(tmp_path / "assistant_cli.toml", opcua_prepared)

    providers = runner.invoke(app, ["assistant", "providers"])
    assert providers.exit_code == 0, providers.output
    assert "llama-cpp" in providers.output

    preflight = runner.invoke(app, ["assistant", "preflight", "--config", str(config_path)])
    assert preflight.exit_code == 0, preflight.output

    run = runner.invoke(
        app,
        [
            "assistant",
            "run",
            "--config",
            str(config_path),
            "--benchmark",
            str(eval_dir),
            "--evidence",
            str(evidence),
            "--out",
            str(tmp_path / "assistant-cli"),
        ],
    )
    assert run.exit_code == 0, run.output
    summarize = runner.invoke(
        app,
        ["assistant", "summarize", "--run", str(tmp_path / "assistant-cli")],
    )
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


def _write_assistant_config(path: Path, prepared: Path) -> Path:
    escaped = str(prepared.resolve()).replace("\\", "\\\\")
    query = THESIS_ASSISTANT_QUERY_TEMPLATE.replace("\\", "\\\\").replace('"', '\\"')
    path.write_text(
        f"""[assistant]
suite_id = "assistant-test"
prepared = "{escaped}"
cases_per_dataset = 2
top_k = 6
minimum_supported_claims = 1
prompt_budget_chars = 8000
query_template = "{query}"

[assistant.provider]
name = "fake"
model = "fake-assistant"
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
    seen_response_formats: list[dict[str, object]] = []

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
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response_format = payload.get("response_format", {})
        if isinstance(response_format, dict) and response_format:
            self.__class__.seen_response_formats.append(dict(response_format))
        content = "- Stubbed provider path is healthy [C1]."
        if isinstance(response_format, dict) and response_format.get("type") in {
            "json_schema",
            "json_object",
        }:
            schema_name = _stub_schema_name(payload, response_format)
            if schema_name == "ClaimEvaluation":
                content = json.dumps(
                    {
                        "is_supported": True,
                        "entailment_label": "entails",
                        "entailment_reasoning": "Stub evidence supports the claim.",
                        "final_disposition": "keep",
                        "rewritten_statement": None,
                    }
                )
            else:
                content = json.dumps(
                    {
                        "symptom_summary": "The event has cited abnormal evidence.",
                        "likely_causes": ["Cited variables indicate the affected area."],
                        "checks": ["Check the cited event variables."],
                        "recommended_actions": ["Preserve cited artifacts."],
                        "escalation_criteria": [],
                    }
                )
        self._json_response(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
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


class _FallbackStructuredStubHandler(_OpenAICompatibleStubHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response_format = payload.get("response_format", {})
        if isinstance(response_format, dict) and response_format:
            self.__class__.seen_response_formats.append(dict(response_format))
        content = "not json"
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            content = json.dumps(
                {
                    "symptom_summary": "The event has cited abnormal evidence.",
                    "likely_causes": [],
                    "checks": ["Check the cited event variables."],
                    "recommended_actions": [],
                    "escalation_criteria": [],
                }
            )
        self._json_response(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


def _stub_schema_name(payload: dict[str, object], response_format: dict[str, object]) -> str:
    nested = response_format.get("json_schema")
    if isinstance(nested, dict):
        name = nested.get("name")
        if isinstance(name, str):
            return name
    messages = payload.get("messages", [])
    if isinstance(messages, list):
        text = "\n".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
        if "ClaimEvaluation" in text or "final_disposition" in text:
            return "ClaimEvaluation"
    return "DraftResponse"


class _InvalidStructuredStubHandler(_OpenAICompatibleStubHandler):
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
                        "message": {"role": "assistant", "content": "not json"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
