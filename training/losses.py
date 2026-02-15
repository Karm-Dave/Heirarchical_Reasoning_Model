# training/losses.py
import torch
<<<<<<< HEAD
import torch.nn.functional as F
import torch.nn as nn
=======
import torch.nn.functional as F
>>>>>>> f7790608f8f7bbb4aa59032a81784b5a16bc8088

<<<<<<< HEAD
IGNORE_LABEL_ID = -100
=======
>>>>>>> f7790608f8f7bbb4aa59032a81784b5a16bc8088

<<<<<<< HEAD
def masked_corss_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID):
    # Compute token-level cross entropy while ignoring padding tokens.
    # Returns per-token loss

    B, T, C = logits.shape

    logits = logits.view(-1, C)
    labels = labels.view(-1)

    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        reduction="none"
    )

    return loss.view(B, T)

class GMHRM(nn.Module):
    """
    Task Loss (masked CE)
    ACT ponder penalty
    Optimal halt supervision
    """
=======
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

>>>>>>> f7790608f8f7bbb4aa59032a81784b5a16bc8088