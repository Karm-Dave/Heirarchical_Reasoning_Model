from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from data.common import PuzzleDatasetMetadata


@dataclass
class Config:
    output_dir: str
    train_size: int
    test_size: int
    seq_len: int
    vocab_size: int
    output_dim: int
    seed: int


@dataclass
class SplitArrays:
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    puzzle_indices: np.ndarray
    group_indices: np.ndarray


def _build_split(cfg: Config, n: int, rng: np.random.Generator) -> SplitArrays:
    inputs = rng.integers(1, cfg.vocab_size + 1, size=(n, cfg.seq_len), dtype=np.int32)
    labels = np.zeros((n, cfg.seq_len), dtype=np.int32)
    labels[:, 0] = rng.integers(1, cfg.output_dim + 1, size=n, dtype=np.int32)

    puzzle_identifiers = np.zeros(n, dtype=np.int32)
    puzzle_indices = np.arange(0, n + 1, dtype=np.int32)
    group_indices = np.arange(0, n + 1, dtype=np.int32)
    return SplitArrays(inputs, labels, puzzle_identifiers, puzzle_indices, group_indices)


def _save_split(cfg: Config, split: str, arrays: SplitArrays):
    split_dir = os.path.join(cfg.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    np.save(os.path.join(split_dir, "all__inputs.npy"), arrays.inputs)
    np.save(os.path.join(split_dir, "all__labels.npy"), arrays.labels)
    np.save(os.path.join(split_dir, "all__puzzle_identifiers.npy"), arrays.puzzle_identifiers)
    np.save(os.path.join(split_dir, "all__puzzle_indices.npy"), arrays.puzzle_indices)
    np.save(os.path.join(split_dir, "all__group_indices.npy"), arrays.group_indices)

    metadata = PuzzleDatasetMetadata(
        seq_len=cfg.seq_len,
        vocab_size=cfg.vocab_size + 1,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(arrays.group_indices) - 1,
        mean_puzzle_examples=1.0,
        sets=["all"],
    )
    with open(os.path.join(split_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f)


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/mock-benchmark")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=81)
    p.add_argument("--vocab-size", type=int, default=12)
    p.add_argument("--output-dim", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Config(
        output_dir=a.output_dir,
        train_size=a.train_size,
        test_size=a.test_size,
        seq_len=a.seq_len,
        vocab_size=a.vocab_size,
        output_dim=a.output_dim,
        seed=a.seed,
    )


def main():
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)
    train = _build_split(cfg, cfg.train_size, rng)
    test = _build_split(cfg, cfg.test_size, rng)

    os.makedirs(cfg.output_dir, exist_ok=True)
    _save_split(cfg, "train", train)
    _save_split(cfg, "test", test)

    with open(os.path.join(cfg.output_dir, "identifiers.json"), "w", encoding="utf-8") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    main()
