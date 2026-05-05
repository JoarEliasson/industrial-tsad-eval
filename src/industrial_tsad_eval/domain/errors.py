"""Domain and application-level exceptions."""

from __future__ import annotations


class IndustrialTSADError(RuntimeError):
    """Base exception for expected toolkit failures."""


class ContractValidationError(IndustrialTSADError):
    """Raised when an artifact violates an expected data contract."""


class PluginNotFoundError(IndustrialTSADError):
    """Raised when a named plugin is not registered."""


class PreparationError(IndustrialTSADError):
    """Raised when raw-to-prepared dataset preparation fails."""


class BenchmarkConfigError(IndustrialTSADError):
    """Raised when a benchmark configuration is invalid."""


class BenchmarkRunError(IndustrialTSADError):
    """Raised when a benchmark run cannot be created or completed."""


class RepositoryError(IndustrialTSADError):
    """Raised when repository-backed artifacts cannot be read or written."""


class OptionalDependencyError(IndustrialTSADError):
    """Raised when an optional dependency is required but unavailable."""


class PreflightError(IndustrialTSADError):
    """Raised when a strict preflight check fails."""


class ProfileRunError(IndustrialTSADError):
    """Raised when a profiling run cannot be created or completed."""
