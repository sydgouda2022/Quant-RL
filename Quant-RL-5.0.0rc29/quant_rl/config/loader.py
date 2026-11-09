"""Strict research configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from quant_rl.config.schema import QuantRLConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    """Fail closed rather than allowing YAML key shadowing."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    result: dict = {}
    for k_node, v_node in node.value:
        key = loader.construct_object(k_node, deep=True)
        if key in result:
            raise ValueError(f"Duplicate YAML configuration key: {key!r} at line {k_node.start_mark.line + 1}")
        result[key] = loader.construct_object(v_node, deep=True)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


class ConfigLoader:
    """Loads and validates `config.yaml` into an immutable `QuantRLConfig`."""

    @staticmethod
    def load(config_path: str | Path | None = None) -> QuantRLConfig:
        if config_path is None:
            root = Path(__file__).resolve().parents[2]
            path = root / "config.yaml"
        else:
            path = Path(config_path).resolve()
            root = path.parent

        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.load(handle, Loader=_UniqueKeyLoader) or {}

        raw["project_root"] = str(root)
        return QuantRLConfig.model_validate(raw)
