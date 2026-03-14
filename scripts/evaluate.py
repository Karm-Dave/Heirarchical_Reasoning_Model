from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pipeline import evaluate_once
from utils.config import load_config


def _default_run_dir(cfg):
    output_root = Path(getattr(cfg.experiment, "output_root", "runs"))
    exp_name = str(getattr(cfg.experiment, "name", "default_run"))
    return output_root / exp_name


def main():
    parser = argparse.ArgumentParser(description="Evaluate GM-HRM pipeline.")
    parser.add_argument("--config", default="configs/main.yaml", help="Path to YAML config.")
    parser.add_argument("--run-dir", default=None, help="Optional explicit run directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    args = parser.parse_args()

    cfg = load_config(args.config, as_namespace=True)
    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(cfg)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    metrics = evaluate_once(cfg, run_dir, checkpoint_path=checkpoint)
    print(
        "eval_ok "
        f"loss={metrics['loss']:.6f} "
        f"acc={metrics['accuracy']:.4f} "
        f"samples={metrics['samples']}"
    )


if __name__ == "__main__":
    main()
