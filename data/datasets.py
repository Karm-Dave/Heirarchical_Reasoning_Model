from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyClassificationDataset(Dataset):
    def __init__(
        self,
        split_dir,
        input_dim: int,
        output_dim: int,
        max_samples: int | None = None,
        normalize_divisor: float = 10.0,
        target_index: int = 0,
    ):
        split_path = Path(split_dir)
        inputs_path = split_path / "all__inputs.npy"
        labels_path = split_path / "all__labels.npy"

        if not inputs_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Missing dataset files in {split_path}. Expected all__inputs.npy and all__labels.npy."
            )

        self.inputs = np.load(inputs_path)
        self.labels = np.load(labels_path)

        if max_samples is not None:
            self.inputs = self.inputs[:max_samples]
            self.labels = self.labels[:max_samples]

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.normalize_divisor = float(normalize_divisor)
        self.target_index = int(target_index)

    def __len__(self):
        return len(self.inputs)

    def _project_input(self, x: np.ndarray) -> np.ndarray:
        # Normalize token ids and adapt to model input dim.
        if self.normalize_divisor > 0:
            x = x.astype(np.float32) / self.normalize_divisor
        else:
            x = x.astype(np.float32)
        if x.shape[0] >= self.input_dim:
            return x[: self.input_dim]
        out = np.zeros(self.input_dim, dtype=np.float32)
        out[: x.shape[0]] = x
        return out

    def __getitem__(self, idx):
        x_raw = self.inputs[idx]
        y_raw = self.labels[idx]

        x = torch.from_numpy(self._project_input(x_raw))
        y = torch.tensor(int(y_raw[self.target_index]) % self.output_dim, dtype=torch.long)
        return x, y


class SudokuNpyDataset(NpyClassificationDataset):
    pass
