"""Provider-agnostic LLM contracts for assistant evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProviderStatus = Literal["ready", "unavailable", "not_configured"]


@dataclass(frozen=True)
class LLMProviderConfig:
    """Runtime configuration for an LLM provider plugin."""

    name: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 60.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data without resolving secrets."""
        return asdict(self)


@dataclass(frozen=True)
class LLMMessage:
    """One chat-style provider message."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to provider-compatible JSON."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral request for assistant generation."""

    messages: list[LLMMessage]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request for artifact capture."""
        return {
            "messages": [message.to_dict() for message in self.messages],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral generation response."""

    text: str
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response for artifact capture."""
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMProviderHealth:
    """Readiness status for one provider configuration."""

    provider: str
    status: ProviderStatus
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return whether the provider is ready for a run."""
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "provider": self.provider,
            "status": self.status,
            "ok": self.ok,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LLMProviderDescription:
    """User-facing provider plugin metadata."""

    name: str
    family: str
    default_model: str
    default_base_url: str | None
    requires_api_key: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)
