"""Domain and application-level exceptions."""

from __future__ import annotations


class IndustrialTSADError(RuntimeError):
    """Base exception for expected toolkit failures."""


class ContractValidationError(IndustrialTSADError):
    """Raised when an artifact violates an expected data contract."""


class PluginNotFoundError(IndustrialTSADError):
    """Raised when a named plugin is not registered."""


class RepositoryError(IndustrialTSADError):
    """Raised when repository-backed artifacts cannot be read or written."""
