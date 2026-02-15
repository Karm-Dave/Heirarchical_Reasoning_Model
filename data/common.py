from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel


class PuzzleDatasetMetadata(BaseModel):
    seq_len: int
    vocab_size: int
    pad_id: int
    ignore_label_id: int
    blank_identifier_id: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    sets: List[str]


def dihedral_transform(arr: np.ndarray, idx: int) -> np.ndarray:
    """
    Apply one of 8 dihedral transforms:
    0..3: rotations by 0/90/180/270
    4..7: horizontal flip + rotations by 0/90/180/270
    """
    idx = int(idx) % 8
    out = np.array(arr, copy=True)

    if idx >= 4:
        out = np.fliplr(out)
        idx -= 4

    if idx > 0:
        out = np.rot90(out, k=idx)

    return out
