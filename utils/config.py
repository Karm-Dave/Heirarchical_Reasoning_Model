# utils/config.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_config(config_dir: str = "configs", model_name: Optional[str] = None):
    cfg_dir = Path(config_dir)

    base_cfg = _read_yaml(cfg_dir / "base.yaml")
    chosen_model = model_name or base_cfg.get("model", {}).get("name", "gm_hrm")

    model_cfg = _read_yaml(cfg_dir / "model" / f"{chosen_model}.yaml")
    train_cfg = _read_yaml(cfg_dir / "train.yaml")
    exp_cfg = _read_yaml(cfg_dir / "experiment.yaml")

    merged = _deep_merge(base_cfg, model_cfg)
    merged = _deep_merge(merged, train_cfg)
    merged = _deep_merge(merged, exp_cfg)

    required = [
        ("model", "name"),
        ("halting", "max_segments"),
        ("memory", "slot_dim"),
        ("training", "lr"),
    ]
    for parent, child in required:
        if parent not in merged or child not in merged[parent]:
            raise KeyError(f"Missing required config key: {parent}.{child}")

    return _to_namespace(merged)
