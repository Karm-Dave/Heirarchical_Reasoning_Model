import torch
import torch.nn as nn


class ACTHalting(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.halt_fc = nn.Linear(dim, 1)

    def forward(self, h_states):
 

        halting_probs = []
        accumulated = None
        expected_steps = 0

        for step, h in enumerate(h_states):

            p = torch.sigmoid(self.halt_fc(h))  # (B, 1)

            if accumulated is None:
                accumulated = p
            else:
                accumulated = accumulated + p

            halting_probs.append(p)

            expected_steps += p

        return halting_probs, expected_steps
