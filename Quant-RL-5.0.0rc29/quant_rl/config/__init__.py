"""Strict Quant-RL configuration with lazy exports to avoid import cycles."""
from __future__ import annotations

from importlib import import_module

_SCHEMA_EXPORTS = {
    "QuantRLConfig", "ResearchConfig", "DataConfig", "IntelligenceConfig",
    "PPOConfig", "OptimizationConfig", "QualificationConfig", "ExecutionConfig",
    "BrokerConnectionConfig", "GeometryModeConfig", "PositionSizingConfig", "LiveConfig",
    "VALID_TIMEFRAMES",
}


def __getattr__(name: str):
    if name in _SCHEMA_EXPORTS:
        return getattr(import_module("quant_rl.config.schema"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SCHEMA_EXPORTS)


__all__ = sorted(_SCHEMA_EXPORTS)
