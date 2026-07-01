"""Config loading.

Extends the lightweight a2l-pr ``ConfigManager.from_yaml`` pattern: plain YAML
parsed into nested dicts, with two additions useful for a multi-stage pipeline:

- ``includes:`` merges sibling YAML files (so the top-level pipeline.yaml can
  pull in dataset/camera/colmap/splat/robot configs);
- dotted access (``cfg.get("dataset.cameras.wrist")``) for terse lookups.

No pydantic / Hydra, matching the sibling project's deliberately light style.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Nested-dict config with ``includes`` merging and dotted lookups."""

    def __init__(self, data: Dict[str, Any] | None = None, source: Path | None = None):
        self.data: Dict[str, Any] = data or {}
        self.source = source

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Merge included sibling configs first, then let this file override.
        merged: Dict[str, Any] = {}
        for inc in data.pop("includes", []) or []:
            inc_path = (path.parent / inc).resolve()
            inc_cfg = cls.from_yaml(inc_path)
            merged = _deep_merge(merged, inc_cfg.data)
        merged = _deep_merge(merged, data)
        return cls(merged, source=path)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        val = self.get(dotted, sentinel)
        if val is sentinel:
            raise KeyError(f"Missing required config key: {dotted!r} (in {self.source})")
        return val

    def section(self, name: str) -> Dict[str, Any]:
        return self.get(name, {}) or {}

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data
