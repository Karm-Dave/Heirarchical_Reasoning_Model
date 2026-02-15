# training/losses.py
import torch
import torch.nn.functional as F


def act_loss(outputs, targets, halting_probs, ponder_cost, expected_steps=None):
    if len(outputs) != len(halting_probs):
        raise ValueError("outputs and halting_probs must have same length.")

    total = 0.0
    for y_hat, p in zip(outputs, halting_probs):
        ce = F.cross_entropy(y_hat, targets, reduction="none")  # (B,)
        w = p.squeeze(-1)  # (B,)
        total = total + (w * ce).mean()

    if expected_steps is None:
        expected_steps = 0.0
        for idx, p in enumerate(halting_probs):
            expected_steps = expected_steps + (idx + 1) * p.mean()

    return total + ponder_cost * expected_steps
