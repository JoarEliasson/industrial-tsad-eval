"""Ports for provider-backed assistant generation."""

from __future__ import annotations

from typing import Protocol

from industrial_tsad_eval.domain.llm import (
    LLMProviderConfig,
    LLMProviderDescription,
    LLMProviderHealth,
    LLMRequest,
    LLMResponse,
)


class LLMProvider(Protocol):
    """Provider-neutral generation interface."""

    @property
    def name(self) -> str:
        """Stable provider name."""

    def healthcheck(self) -> LLMProviderHealth:
        """Check whether the provider can serve requests."""

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for a chat-style request."""


class LLMProviderPlugin(Protocol):
    """Factory and metadata interface for LLM provider plugins."""

    @property
    def name(self) -> str:
        """Stable provider plugin name."""

    def describe(self) -> LLMProviderDescription:
        """Return provider metadata for discovery."""

    def default_config(self) -> LLMProviderConfig:
        """Return a safe default configuration."""

    def create(self, config: LLMProviderConfig) -> LLMProvider:
        """Create a provider instance."""
