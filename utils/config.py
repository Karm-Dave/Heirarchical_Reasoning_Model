from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_extended_config(path: Path) -> Dict[str, Any]:
    cfg = _read_yaml(path)
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg

    parent_path = (path.parent / parent).resolve()
    parent_cfg = _resolve_extended_config(parent_path)
    return deep_merge_dict(parent_cfg, cfg)


def _validate(cfg: Dict[str, Any]):
    required = [
        ("seed",),
        ("device",),
        ("model", "name"),
        ("model", "hidden_dim"),
        ("model", "input_dim"),
        ("model", "output_dim"),
        ("training", "batch_size"),
        ("training", "lr"),
        ("training", "epochs"),
        ("data", "root"),
    ]
    for key_path in required:
        node: Any = cfg
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                joined = ".".join(key_path)
                raise KeyError(f"Missing required config key: {joined}")
            node = node[key]


def load_config(config_path: str = "configs/main.yaml", as_namespace: bool = True):
    path = Path(config_path).resolve()
    cfg = _resolve_extended_config(path)
    _validate(cfg)
    return _to_namespace(cfg) if as_namespace else cfg
