from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from data.datasets import NpyClassificationDataset
from models.gm_hrm import GMHRM
from training.evaluation import evaluate
from training.trainer import Trainer


def resolve_device(cfg) -> torch.device:
    requested = str(cfg.device).lower()
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_optimizer(model: torch.nn.Module, cfg):
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))
    name = str(getattr(cfg.optimizer, "name", "adam")).lower()

    if name == "adamw":
        betas = tuple(getattr(cfg.optimizer, "betas", [0.9, 0.999]))
        eps = float(getattr(cfg.optimizer, "eps", 1e-8))
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_model(cfg):
    model_name = str(cfg.model.name).lower()
    if model_name != "gm_hrm":
        raise ValueError(f"Unsupported model.name '{cfg.model.name}'. Only 'gm_hrm' is implemented.")
    return GMHRM(cfg)


def _make_loader(cfg, split: str, max_samples: int | None, shuffle: bool):
    split_dir = Path(cfg.data.root) / split
    dataset = NpyClassificationDataset(
        split_dir=split_dir,
        input_dim=int(cfg.model.input_dim),
        output_dim=int(cfg.model.output_dim),
        max_samples=max_samples,
        normalize_divisor=float(getattr(cfg.data, "normalize_divisor", 10.0)),
        target_index=int(getattr(cfg.data, "target_index", 0)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=shuffle,
        num_workers=int(getattr(cfg.data, "num_workers", 0)),
        pin_memory=bool(getattr(cfg.data, "pin_memory", True)),
    )


def _checkpoint_path(cfg, run_dir: Path) -> Path:
    name = str(getattr(cfg.experiment, "checkpoint_name", "model.pt"))
    return run_dir / name


def train_once(cfg, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.seed))
    device = resolve_device(cfg)

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    trainer = Trainer(model, optimizer, cfg)
    trainer.train()

    max_train_samples = getattr(cfg.data, "max_train_samples", None)
    train_loader = _make_loader(
        cfg=cfg,
        split=str(getattr(cfg.data, "train_split", "train")),
        max_samples=int(max_train_samples) if max_train_samples is not None else None,
        shuffle=True,
    )

    epochs = int(cfg.training.epochs)
    max_batches = getattr(cfg.training, "max_batches_per_epoch", None)
    max_batches = int(max_batches) if max_batches is not None else None

    step_count = 0
    last_loss = None
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            last_loss = trainer.train_step((x, y))
            step_count += 1
            print(f"train epoch={epoch+1} batch={batch_idx+1} loss={last_loss:.6f}")

    if last_loss is None:
        raise RuntimeError("No training steps were executed. Check dataset and config.")

    ckpt_path = _checkpoint_path(cfg, run_dir)
    if bool(getattr(cfg.experiment, "save_checkpoint", True)):
        torch.save({"model": model.state_dict()}, ckpt_path)

    return {
        "device": str(device),
        "steps": step_count,
        "last_loss": float(last_loss),
        "checkpoint_path": str(ckpt_path),
    }


def evaluate_once(cfg, run_dir: Path, checkpoint_path: Path | None = None) -> Dict[str, Any]:
    device = resolve_device(cfg)
    model = build_model(cfg).to(device)

    ckpt_path = checkpoint_path if checkpoint_path is not None else _checkpoint_path(cfg, run_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    max_test_samples = getattr(cfg.data, "max_test_samples", None)
    test_loader = _make_loader(
        cfg=cfg,
        split=str(getattr(cfg.data, "test_split", "test")),
        max_samples=int(max_test_samples) if max_test_samples is not None else None,
        shuffle=False,
    )

    max_batches = getattr(cfg.evaluation, "max_batches", None)
    if max_batches is not None:
        max_batches = int(max_batches)
        # lightweight wrapper for bounded iteration
        bounded = []
        for i, batch in enumerate(test_loader):
            if i >= max_batches:
                break
            bounded.append(batch)
        metrics = evaluate(model, bounded, device, cfg)
    else:
        metrics = evaluate(model, test_loader, device, cfg)

    return metrics


def namespace_from_dict(d: Dict[str, Any]):
    def _to_ns(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_ns(v) for v in obj]
        return obj

    return _to_ns(d)
