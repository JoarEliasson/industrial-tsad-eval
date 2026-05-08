from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from typer.testing import CliRunner

from industrial_tsad_eval.application.assistant_replay import (
    RunAssistantReplaySuite,
    _claims_from_draft,
    _referee_request_for_claim,
)
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
from industrial_tsad_eval.domain.operator import (
    OperatorEvidenceHit,
    OperatorQuery,
    OperatorRetrievalResult,
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


def test_check_tag_citation_selection_prefers_top_variable_exact_match():
    retrieval = _tag_retrieval_result("Plant/TEP/XMV_09")
    draft = DraftResponse(checks=["Check Plant/TEP/XMV_09."])

    claims = _claims_from_draft(draft, retrieval)

    assert claims[0].statement == "Check Plant/TEP/XMV_09."
    assert claims[0].cited_evidence_ids[:2] == ["C1", "C5"]
    assert "C2" not in claims[0].cited_evidence_ids


def test_referee_request_includes_supporting_facts_for_exact_tag_checks():
    retrieval = _tag_retrieval_result("Plant/HAI/P1/FT02")
    claim = _claims_from_draft(
        DraftResponse(checks=["Check Plant/HAI/P1/FT02"]),
        retrieval,
    )[0]
    request = _referee_request_for_claim(
        case=_assistant_case("HAI", "hai-event"),
        claim=claim,
        retrieval=retrieval,
        config=AssistantReplayConfig(
            suite_id="supporting-facts",
            prepared="prepared/HAI",
            provider=LLMProviderConfig(name="fake", model="fake-assistant"),
        ),
    )
    payload = json.loads(request.messages[1].content)

    assert payload["supporting_facts"]["bounded_variable_inspection"] is True
    assert payload["supporting_facts"]["check_target"] == "plant/hai/p1/ft02"
    assert payload["supporting_facts"]["exact_top_variable_citations"] == ["C1"]


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
            "--no-progress",
        ],
    )
    assert run.exit_code == 0, run.output
    run_root = tmp_path / "repro" / "smoke"
    assert (run_root / "benchmark" / "summary.json").exists()
    assert (run_root / "assistant" / "assistant_summary.json").exists()
    assert (run_root / "summaries" / "thesis_crosswalk.md").exists()
    assert (run_root / "summaries" / "detection_tables.csv").exists()
    assert (run_root / "summaries" / "explanation_results.csv").exists()
    assert (run_root / "summaries" / "explanation_results_split_summary.csv").exists()
    assert (run_root / "summaries" / "assistant_faithfulness_logs.csv").exists()
    assert (run_root / "summaries" / "profiling_logs.csv").exists()
    assert (run_root / "summaries" / "dataset_splits.json").exists()
    assert (run_root / "summaries" / "hyperparameters.toml").exists()
    assert (run_root / "summaries" / "scoring_config.json").exists()
    assert (run_root / "summaries" / "planner_prompt.txt").exists()
    assert (run_root / "summaries" / "referee_prompt.txt").exists()
    assert (
        run_root / "summaries" / "assistant" / "schemas" / "draft_response.schema.json"
    ).exists()
    assert (run_root / "progress_snapshot.json").exists()
    assert "Return JSON matching DraftResponse only." in (
        run_root / "summaries" / "planner_prompt.txt"
    ).read_text(encoding="utf-8")
    claim_schema = json.loads(
        (
            run_root / "summaries" / "assistant" / "schemas" / "claim_evaluation.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert claim_schema["title"] == "ClaimEvaluation"
    status = runner.invoke(app, ["reproduce", "status", "--run", str(run_root)])
    assert status.exit_code == 0, status.output
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True


def test_reproduction_cli_writes_verification_profile(tmp_path: Path):
    config_path = tmp_path / "verification.toml"

    result = runner.invoke(
        app,
        [
            "reproduce",
            "init-config",
            "--out",
            str(config_path),
            "--profile",
            "thesis-verification",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "forecast-lstm-tiny" in config_path.read_text(encoding="utf-8")


def test_thesis_full_profile_matches_current_draft_parameters(tmp_path: Path):
    config_path = tmp_path / "full.toml"
    write_default_reproduction_config(config_path, profile="thesis-full")

    config = load_reproduction_config(config_path)
    experiments = {item.experiment_id: item for item in config.benchmark.experiments()}

    assert config.evidence_sources == ["oracle", "operational"]
    assert config.assistant_evidence_source == "operational"
    assert config.profile_experiment_limit is None
    assert [detector.id for detector in config.benchmark.detectors] == [
        "forecast-ridge",
        "dra",
        "interfusion",
        "drcad",
    ]
    assert experiments["TEP__dra__naive"].detector.parameters["window"] == 100
    assert experiments["TEP__dra__naive"].detector.parameters["train_stride"] == 5
    assert experiments["SWaT__dra__all_in_one"].detector.parameters["train_stride"] == 10
    assert experiments["SWaT__dra__all_in_one"].detector.parameters["score_stride"] == 1
    assert experiments["SWaT__interfusion__naive"].detector.parameters["latent_dim"] == 3
    assert experiments["SWaT__interfusion__naive"].detector.parameters["kl_warmup"] == 10
    assert experiments["HAI-CPPS__drcad__zero_shot"].detector.parameters["patch_size"] == 5
    assert experiments["HAI-CPPS__drcad__zero_shot"].detector.parameters["lr"] == 0.0001


def test_thesis_full_profile_matches_current_draft_scoring_policy(tmp_path: Path):
    config_path = tmp_path / "full.toml"
    write_default_reproduction_config(config_path, profile="thesis-full")
    config = load_reproduction_config(config_path)

    default_policy = config.benchmark.evaluation.policy_for("SWaT", "naive")
    tep_policy = config.benchmark.evaluation.policy_for("TEP", "naive")
    hai_policy = config.benchmark.evaluation.policy_for("HAI", "all_in_one")
    cpps_policy = config.benchmark.evaluation.policy_for("HAI-CPPS", "zero_shot")

    assert default_policy.merge_gap_s == 10.0
    assert default_policy.grace_s == 5.0
    assert tep_policy.merge_gap_mode == "auto_period"
    assert tep_policy.merge_gap_skipped_samples == 1
    assert tep_policy.merge_gap_jitter_ratio == 0.1
    assert hai_policy.merge_gap_s == 30.0
    assert hai_policy.grace_s == 120.0
    assert cpps_policy.grace_s == 6.0


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
            "--no-progress",
        ],
    )
    assert run.exit_code == 0, run.output
    assert (tmp_path / "assistant-cli" / "progress_snapshot.json").exists()
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


def _assistant_case(dataset: str, event_id: str):
    from industrial_tsad_eval.domain.assistant_replay import AssistantCase

    return AssistantCase(
        case_id=f"{dataset}__{event_id}",
        dataset=dataset,
        run_id=f"{dataset.lower()}/test/run_001",
        event_id=event_id,
        query=f"Check {event_id}",
        top_variables=[],
        evidence_relative_path="bundles/run/event/evidence.json",
        expected_retrieval_event_ids=[event_id],
    )


def _tag_retrieval_result(tag: str) -> OperatorRetrievalResult:
    return OperatorRetrievalResult(
        query=OperatorQuery(query=f"Check {tag}.", dataset="fixture", event_id="event-1"),
        hits=[
            OperatorEvidenceHit(
                source_id="evidence::event-1::top_variables",
                source_type="evidence_bundle",
                title="Evidence Bundle: event-1",
                role="top_variables",
                rank=1,
                score=1.0,
                text=f"Top ranked variables for event event-1: {tag}, Plant/FIXTURE/OTHER.",
                citation_id="C1",
                dataset="fixture",
                event_id="event-1",
                run_id="fixture/test/run_001",
                metadata={"top_variables": [tag, "Plant/FIXTURE/OTHER"]},
            ),
            OperatorEvidenceHit(
                source_id="evidence::event-1::score_context",
                source_type="evidence_bundle",
                title="Evidence Bundle: event-1",
                role="score_context",
                rank=2,
                score=2.0,
                text="Score context for event event-1: high score.",
                citation_id="C2",
                dataset="fixture",
                event_id="event-1",
                run_id="fixture/test/run_001",
            ),
            OperatorEvidenceHit(
                source_id="evidence::event-1::local_rankings",
                source_type="evidence_bundle",
                title="Evidence Bundle: event-1",
                role="local_rankings",
                rank=5,
                score=0.5,
                text=f"Local variable rankings for event event-1 include {tag}.",
                citation_id="C5",
                dataset="fixture",
                event_id="event-1",
                run_id="fixture/test/run_001",
            ),
        ],
    )


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
