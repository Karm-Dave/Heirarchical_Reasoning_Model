from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.pipeline import evaluate_once, namespace_from_dict, train_once
from utils.config import deep_merge_dict, load_config


def _dataset_ready(data_root: Path, train_split: str, test_split: str) -> bool:
    train_inputs = data_root / train_split / "all__inputs.npy"
    test_inputs = data_root / test_split / "all__inputs.npy"
    return train_inputs.exists() and test_inputs.exists()


def _maybe_generate_dataset(
    run_cfg: Dict[str, Any],
    benchmark_name: str,
    level_name: str,
    generator_cfg: Dict[str, Any] | None,
):
    data_root = Path(run_cfg["data"]["root"])
    train_split = str(run_cfg["data"].get("train_split", "train"))
    test_split = str(run_cfg["data"].get("test_split", "test"))

    if _dataset_ready(data_root, train_split, test_split):
        return

    if not bool(run_cfg["data"].get("auto_generate", False)):
        raise FileNotFoundError(
            f"Dataset missing for '{benchmark_name}/{level_name}' at '{data_root}', and auto_generate is disabled."
        )

    if not isinstance(generator_cfg, dict) or "module" not in generator_cfg:
        raise FileNotFoundError(
            f"Dataset missing for '{benchmark_name}/{level_name}' at '{data_root}', and no generator is configured."
        )

    cmd = [sys.executable, "-m", str(generator_cfg["module"])]
    cmd.extend([str(x) for x in generator_cfg.get("args", [])])
    print(f"[data] generating '{benchmark_name}/{level_name}': {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not _dataset_ready(data_root, train_split, test_split):
        raise FileNotFoundError(
            f"Generator completed but dataset still missing at '{data_root}' for '{benchmark_name}/{level_name}'."
        )


def _iter_levels(base_cfg: Dict[str, Any], benchmark_name: str):
    definitions = base_cfg["benchmarks"]["definitions"]
    if benchmark_name not in definitions:
        raise KeyError(f"Missing benchmark definition: {benchmark_name}")

    bench_def = definitions[benchmark_name]
    levels = bench_def.get("levels", [])
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"Benchmark '{benchmark_name}' has no levels configured.")

    for level in levels:
        if "name" not in level:
            raise ValueError(f"Benchmark '{benchmark_name}' has a level without 'name'.")

        level_name = str(level["name"])
        dataset_cfg = level.get("dataset", {})
        if "root" not in dataset_cfg:
            raise ValueError(f"Benchmark '{benchmark_name}/{level_name}' missing dataset.root")

        overrides = {
            "data": {
                "benchmark": benchmark_name,
                "root": dataset_cfg["root"],
                "normalize_divisor": dataset_cfg.get(
                    "normalize_divisor", base_cfg["data"].get("normalize_divisor", 10.0)
                ),
                "target_index": dataset_cfg.get(
                    "target_index", base_cfg["data"].get("target_index", 0)
                ),
            },
            "model": {
                "input_dim": dataset_cfg.get("input_dim", base_cfg["model"]["input_dim"]),
                "output_dim": dataset_cfg.get("output_dim", base_cfg["model"]["output_dim"]),
            },
        }

        run_cfg = deep_merge_dict(base_cfg, overrides)
        if "overrides" in level:
            run_cfg = deep_merge_dict(run_cfg, level["overrides"])

        generator = level.get("generator")
        yield level_name, run_cfg, generator


def main():
    parser = argparse.ArgumentParser(description="Run benchmark + ablation experiments.")
    parser.add_argument("--config", default="configs/main.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    cfg = load_config(args.config, as_namespace=False)
    benchmark_names = cfg["benchmarks"]["enabled"]
    if not isinstance(benchmark_names, list):
        raise ValueError("benchmarks.enabled must be a list.")

    variant_specs: List[Dict[str, Any]]
    if bool(cfg.get("ablations", {}).get("enabled", False)):
        variant_specs = cfg["ablations"]["variants"]
    else:
        variant_specs = [{"name": "default", "overrides": {}}]

    all_results = []
    for benchmark_name in benchmark_names:
        print(f"\n=== benchmark: {benchmark_name} ===")
        for level_name, benchmark_base, generator in _iter_levels(cfg, benchmark_name):
            print(f"\n== level: {level_name} ==")
            _maybe_generate_dataset(benchmark_base, benchmark_name, level_name, generator)

            for variant in variant_specs:
                variant_name = str(variant["name"])
                print(f"\n--- variant: {variant_name} ---")

                run_cfg_dict = deep_merge_dict(benchmark_base, variant.get("overrides", {}))
                run_cfg_dict["experiment"]["name"] = (
                    f"{cfg['experiment']['name']}/{benchmark_name}/{level_name}/{variant_name}"
                )
                run_cfg = namespace_from_dict(run_cfg_dict)

                run_dir = Path(run_cfg.experiment.output_root) / run_cfg.experiment.name
                train_result = train_once(run_cfg, run_dir)
                eval_result = evaluate_once(run_cfg, run_dir)

                result = {
                    "benchmark": benchmark_name,
                    "level": level_name,
                    "variant": variant_name,
                    "run_dir": str(run_dir),
                    "device": train_result["device"],
                    "steps": train_result["steps"],
                    "train_last_loss": train_result["last_loss"],
                    "eval_loss": eval_result["loss"],
                    "eval_accuracy": eval_result["accuracy"],
                    "eval_samples": eval_result["samples"],
                }
                all_results.append(result)
                print(
                    "result "
                    f"device={result['device']} "
                    f"steps={result['steps']} "
                    f"train_last_loss={result['train_last_loss']:.6f} "
                    f"eval_loss={result['eval_loss']:.6f} "
                    f"eval_acc={result['eval_accuracy']:.4f}"
                )

    output_root = Path(cfg["experiment"].get("output_root", "runs"))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"{cfg['experiment']['name']}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nsummary_saved {summary_path}")


if __name__ == "__main__":
    main()
