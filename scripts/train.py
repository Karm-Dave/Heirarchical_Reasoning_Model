from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import torch
from torch.utils.data import DataLoader
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.datasets import SudokuNpyDataset
from models.gm_hrm import GMHRM
from training.trainer import Trainer


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_base_config():
    cfg_path = REPO_ROOT / "configs" / "base.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _to_namespace(cfg)


def build_optimizer(model: torch.nn.Module, cfg):
    lr = float(cfg.training.lr)
    weight_decay = float(getattr(cfg.training, "weight_decay", 0.0))
    name = getattr(getattr(cfg, "optimizer", SimpleNamespace(name="adam")), "name", "adam").lower()

    if name == "adamw":
        betas = tuple(getattr(cfg.optimizer, "betas", [0.9, 0.999]))
        eps = float(getattr(cfg.optimizer, "eps", 1e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def main():
    cfg = load_base_config()

    requested_device = str(cfg.device).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    torch.manual_seed(int(cfg.seed))

    model = GMHRM(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    trainer = Trainer(model, optimizer, cfg)
    trainer.train()

    data_root = Path("data/sudoku-mini")
    train_dir = data_root / "train"
    run_dir = Path("runs/minimal")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = SudokuNpyDataset(
        split_dir=train_dir,
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        max_samples=3,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    last_loss = None
    max_batches = 3
    for epoch in range(int(cfg.training.epochs)):
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            last_loss = trainer.train_step((x, y))
            print(f"train epoch={epoch+1} batch={batch_idx+1} loss={last_loss:.6f}")

    if last_loss is None:
        raise RuntimeError("No training steps were executed. Check dataset files.")

    ckpt_path = run_dir / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"train_ok last_loss={last_loss:.6f} ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
