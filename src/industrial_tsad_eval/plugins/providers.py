"""LLM provider plugins for thesis-style assistant replay."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from industrial_tsad_eval.domain.errors import PluginNotFoundError, ProviderConfigError
from industrial_tsad_eval.domain.llm import (
    LLMProviderConfig,
    LLMProviderDescription,
    LLMProviderHealth,
    LLMRequest,
    LLMResponse,
)
from industrial_tsad_eval.ports.llm import LLMProvider, LLMProviderPlugin

CITATION_RE = re.compile(r"\[(C\d+)\]")


@dataclass
class LLMProviderRegistry:
    """In-memory registry for LLM provider plugins."""

    _plugins: dict[str, LLMProviderPlugin] = field(default_factory=dict)

    def register(self, plugin: LLMProviderPlugin) -> None:
        """Register or replace a provider plugin."""
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> LLMProviderPlugin:
        """Return a provider plugin by name."""
        try:
            return self._plugins[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._plugins)) or "<none>"
            raise PluginNotFoundError(
                f"Unknown LLM provider {name!r}. Available providers: {available}."
            ) from exc

    def names(self) -> list[str]:
        """Return registered provider names."""
        return sorted(self._plugins)


class FakeProviderPlugin:
    """Deterministic provider for tests, smoke runs, and CI."""

    name = "fake"

    def describe(self) -> LLMProviderDescription:
        """Return provider metadata."""
        return LLMProviderDescription(
            name=self.name,
            family="deterministic",
            default_model="fake-rq3",
            default_base_url=None,
            requires_api_key=False,
            description="Deterministic provider for contract tests and smoke reproduction.",
        )

    def default_config(self) -> LLMProviderConfig:
        """Return a safe fake-provider config."""
        return LLMProviderConfig(name=self.name, model="fake-rq3")

    def create(self, config: LLMProviderConfig) -> LLMProvider:
        """Create a deterministic provider."""
        return FakeProvider(config)


class FakeProvider:
    """Deterministic provider that emits cited assistant claims."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    @property
    def name(self) -> str:
        """Stable provider name."""
        return self.config.name

    def healthcheck(self) -> LLMProviderHealth:
        """Return ready status."""
        return LLMProviderHealth(self.name, "ready", "Deterministic fake provider is ready.")

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a deterministic response grounded in prompt citations."""
        prompt = "\n".join(message.content for message in request.messages)
        citations = sorted(set(CITATION_RE.findall(prompt)))
        if not citations:
            text = "I must abstain because no cited evidence was provided."
        else:
            first = citations[0]
            second = citations[1] if len(citations) > 1 else first
            text = "\n".join(
                [
                    f"- Likely causes should be assessed from the ranked evidence [{first}].",
                    f"- First checks should compare the event window and top variables [{second}].",
                    (
                        "- Immediate actions should preserve artifacts before process "
                        f"changes [{first}]."
                    ),
                ]
            )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.config.model,
            metadata={"deterministic": True, "citation_count": len(citations)},
        )


class OpenAICompatibleProviderPlugin:
    """Provider plugin for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        *,
        name: str,
        family: str,
        default_model: str,
        default_base_url: str | None,
        default_api_key_env: str | None,
        description: str,
        requires_api_key: bool = True,
    ):
        self.name = name
        self.family = family
        self.default_model = default_model
        self.default_base_url = default_base_url
        self.default_api_key_env = default_api_key_env
        self.description = description
        self.requires_api_key = requires_api_key

    def describe(self) -> LLMProviderDescription:
        """Return provider metadata."""
        return LLMProviderDescription(
            name=self.name,
            family=self.family,
            default_model=self.default_model,
            default_base_url=self.default_base_url,
            requires_api_key=self.requires_api_key,
            description=self.description,
        )

    def default_config(self) -> LLMProviderConfig:
        """Return a default config."""
        return LLMProviderConfig(
            name=self.name,
            model=self.default_model,
            base_url=self.default_base_url,
            api_key_env=self.default_api_key_env,
        )

    def create(self, config: LLMProviderConfig) -> LLMProvider:
        """Create an OpenAI-compatible provider."""
        merged = _merge_defaults(config, self.default_config())
        if merged.base_url is None:
            raise ProviderConfigError(f"Provider {self.name!r} requires base_url.")
        return OpenAICompatibleProvider(merged, requires_api_key=self.requires_api_key)


class OpenAICompatibleProvider:
    """Minimal HTTP adapter for OpenAI-compatible chat APIs."""

    def __init__(self, config: LLMProviderConfig, *, requires_api_key: bool):
        self.config = config
        self.requires_api_key = requires_api_key

    @property
    def name(self) -> str:
        """Stable provider name."""
        return self.config.name

    def healthcheck(self) -> LLMProviderHealth:
        """Check static config and optionally a local model endpoint."""
        api_key = _api_key(self.config)
        if self.requires_api_key and api_key is None:
            return LLMProviderHealth(
                self.name,
                "not_configured",
                f"Missing API key environment variable {self.config.api_key_env!r}.",
            )
        if self.config.name == "llama-cpp" or self.config.extra.get("healthcheck"):
            return _http_healthcheck(self.config, api_key)
        return LLMProviderHealth(
            self.name,
            "ready",
            "Provider configuration is present; live network healthcheck was not requested.",
            {"base_url": self.config.base_url},
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call an OpenAI-compatible chat-completions endpoint."""
        api_key = _api_key(self.config)
        if self.requires_api_key and api_key is None:
            raise ProviderConfigError(
                f"Missing API key environment variable {self.config.api_key_env!r}."
            )
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_dict() for message in request.messages],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        payload.update(self.config.extra.get("request_overrides", {}))
        response = _post_json(
            _join_url(str(self.config.base_url), "chat/completions"),
            payload,
            timeout_s=self.config.timeout_s,
            api_key=api_key,
        )
        choices = response.get("choices", [])
        text = ""
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message", {})
                if isinstance(message, dict):
                    text = str(message.get("content", ""))
        return LLMResponse(
            text=text,
            provider=self.name,
            model=self.config.model,
            metadata={"raw_response_keys": sorted(response.keys())},
        )


class AnthropicProviderPlugin:
    """Provider plugin for Anthropic Messages API."""

    name = "anthropic"

    def describe(self) -> LLMProviderDescription:
        """Return provider metadata."""
        return LLMProviderDescription(
            name=self.name,
            family="anthropic",
            default_model="claude-3-5-sonnet-latest",
            default_base_url="https://api.anthropic.com/v1",
            requires_api_key=True,
            description="Anthropic Messages API provider using an environment API key.",
        )

    def default_config(self) -> LLMProviderConfig:
        """Return a default Anthropic config."""
        return LLMProviderConfig(
            name=self.name,
            model="claude-3-5-sonnet-latest",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
        )

    def create(self, config: LLMProviderConfig) -> LLMProvider:
        """Create an Anthropic provider."""
        return AnthropicProvider(_merge_defaults(config, self.default_config()))


class AnthropicProvider:
    """Minimal HTTP adapter for Anthropic Messages API."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    @property
    def name(self) -> str:
        """Stable provider name."""
        return self.config.name

    def healthcheck(self) -> LLMProviderHealth:
        """Check static provider configuration."""
        if _api_key(self.config) is None:
            return LLMProviderHealth(
                self.name,
                "not_configured",
                f"Missing API key environment variable {self.config.api_key_env!r}.",
            )
        return LLMProviderHealth(
            self.name,
            "ready",
            "Provider configuration is present; live network healthcheck was not requested.",
            {"base_url": self.config.base_url},
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic Messages API."""
        api_key = _api_key(self.config)
        if api_key is None:
            raise ProviderConfigError(
                f"Missing API key environment variable {self.config.api_key_env!r}."
            )
        system_messages = [item.content for item in request.messages if item.role == "system"]
        chat_messages = [
            item.to_dict() for item in request.messages if item.role in {"user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": chat_messages,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        payload.update(self.config.extra.get("request_overrides", {}))
        response = _post_json(
            _join_url(str(self.config.base_url), "messages"),
            payload,
            timeout_s=self.config.timeout_s,
            api_key=api_key,
            headers={
                "anthropic-version": str(self.config.extra.get("anthropic_version", "2023-06-01"))
            },
        )
        content = response.get("content", [])
        text_parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
        return LLMResponse(
            text="\n".join(text_parts),
            provider=self.name,
            model=self.config.model,
            metadata={"raw_response_keys": sorted(response.keys())},
        )


def default_llm_provider_registry() -> LLMProviderRegistry:
    """Create the default provider registry."""
    registry = LLMProviderRegistry()
    registry.register(FakeProviderPlugin())
    registry.register(
        OpenAICompatibleProviderPlugin(
            name="llama-cpp",
            family="local-openai-compatible",
            default_model="local-llama",
            default_base_url="http://127.0.0.1:8080/v1",
            default_api_key_env=None,
            requires_api_key=False,
            description="Local llama.cpp OpenAI-compatible chat server.",
        )
    )
    registry.register(
        OpenAICompatibleProviderPlugin(
            name="openai-compatible",
            family="openai-compatible",
            default_model="model",
            default_base_url=None,
            default_api_key_env=None,
            requires_api_key=False,
            description="Generic OpenAI-compatible endpoint for local or cloud providers.",
        )
    )
    registry.register(
        OpenAICompatibleProviderPlugin(
            name="openai",
            family="openai",
            default_model="gpt-4.1-mini",
            default_base_url="https://api.openai.com/v1",
            default_api_key_env="OPENAI_API_KEY",
            description="OpenAI chat-completions compatible provider.",
        )
    )
    registry.register(
        OpenAICompatibleProviderPlugin(
            name="google",
            family="google",
            default_model="gemini-2.0-flash",
            default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            default_api_key_env="GOOGLE_API_KEY",
            description="Google Gemini OpenAI-compatible endpoint.",
        )
    )
    registry.register(
        OpenAICompatibleProviderPlugin(
            name="xai",
            family="xai",
            default_model="grok-3-mini",
            default_base_url="https://api.x.ai/v1",
            default_api_key_env="XAI_API_KEY",
            description="xAI OpenAI-compatible endpoint.",
        )
    )
    registry.register(AnthropicProviderPlugin())
    return registry


def _merge_defaults(config: LLMProviderConfig, defaults: LLMProviderConfig) -> LLMProviderConfig:
    return LLMProviderConfig(
        name=config.name,
        model=config.model or defaults.model,
        base_url=config.base_url or defaults.base_url,
        api_key_env=config.api_key_env or defaults.api_key_env,
        timeout_s=config.timeout_s,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        seed=config.seed,
        extra=dict(config.extra),
    )


def _api_key(config: LLMProviderConfig) -> str | None:
    if config.api_key_env is None:
        return None
    value = os.environ.get(config.api_key_env)
    return value if value else None


def _http_healthcheck(config: LLMProviderConfig, api_key: str | None) -> LLMProviderHealth:
    try:
        url = _join_url(str(config.base_url), "models")
        request = urllib.request.Request(url, method="GET")
        if api_key is not None:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=min(config.timeout_s, 2.0)) as response:
            return LLMProviderHealth(
                config.name,
                "ready",
                "Provider endpoint responded.",
                {"status_code": response.status, "url": url},
            )
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return LLMProviderHealth(
            config.name,
            "unavailable",
            f"Provider endpoint is unavailable: {type(exc).__name__}: {exc}",
            {"base_url": config.base_url},
        )


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **dict(headers or {})},
    )
    if api_key is not None:
        request.add_header("Authorization", f"Bearer {api_key}")
        request.add_header("x-api-key", api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            decoded = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderConfigError(
            f"Provider request failed with HTTP {exc.code}: {detail}"
        ) from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise ProviderConfigError(f"Provider request failed: {type(exc).__name__}: {exc}") from exc
    payload_obj = json.loads(decoded)
    if not isinstance(payload_obj, dict):
        raise ProviderConfigError("Provider response was not a JSON object.")
    return payload_obj


def _join_url(base_url: str, suffix: str) -> str:
    return base_url.rstrip("/") + "/" + suffix.lstrip("/")
