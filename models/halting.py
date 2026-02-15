# models/halting.py
import torch
import torch.nn as nn


class ACTHalting(nn.Module):
    def __init__(self, dim: int, max_segments: int, threshold: float = 1.0):
        super().__init__()
        self.halt_fc = nn.Linear(dim, 1)
        self.max_segments = max_segments
        self.threshold = threshold

    def forward(self, h_states):
        if len(h_states) == 0:
            raise ValueError("h_states cannot be empty.")

        halting_probs = []
        batch_size = h_states[0].shape[0]
        device = h_states[0].device
        remaining = torch.ones(batch_size, 1, device=device)

        for step, h in enumerate(h_states[: self.max_segments]):
            p = torch.sigmoid(self.halt_fc(h))
            p = torch.minimum(p, remaining)
            halting_probs.append(p)
            remaining = remaining - p

            if self.threshold is not None:
                halted_mass = 1.0 - remaining
                if torch.all(halted_mass >= self.threshold):
                    break

        if halting_probs:
            halting_probs[-1] = halting_probs[-1] + remaining

        expected_steps = 0.0
        for idx, p in enumerate(halting_probs):
            expected_steps = expected_steps + (idx + 1) * p
        expected_steps = expected_steps.mean()

        return halting_probs, expected_steps
