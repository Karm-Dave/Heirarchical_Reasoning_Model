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
from training.evaluation import evaluate


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


def main():
    cfg = load_base_config()

    requested_device = str(cfg.device).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    data_root = Path("data/sudoku-mini")
    test_dir = data_root / "test"
    ckpt_path = Path("runs/minimal/model.pt")

    dataset = SudokuNpyDataset(
        split_dir=test_dir,
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        max_samples=3,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    model = GMHRM(cfg).to(device)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    metrics = evaluate(model, dataloader, device, cfg)
    print(
        f"eval_ok loss={metrics['loss']:.6f} "
        f"acc={metrics['accuracy']:.4f} samples={metrics['samples']}"
    )


if __name__ == "__main__":
    main()
