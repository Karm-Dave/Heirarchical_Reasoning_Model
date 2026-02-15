# training/trainer.py
import torch.nn as nn

from training.losses import act_loss


class Trainer(nn.Module):
    def __init__(self, model, optimizer, cfg):
        super().__init__()
        self.model = model
        self.opt = optimizer
        self.cfg = cfg

    def train_step(self, batch):
        x, y = batch
        outputs, h_states = self.model(x)

        if self.cfg.halting.enabled and self.model.halting is not None:
            halting_probs, expected_steps = self.model.halting(h_states)
        else:
            halting_probs = [o.new_zeros((o.size(0), 1)) for o in outputs]
            halting_probs[-1] = halting_probs[-1] + 1.0
            expected_steps = None

        loss = act_loss(
            outputs=outputs,
            targets=y,
            halting_probs=halting_probs,
            ponder_cost=self.cfg.halting.ponder_cost,
            expected_steps=expected_steps,
        )

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return float(loss.item())
