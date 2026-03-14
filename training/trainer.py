# training/trainer.py
import torch.nn as nn
import torch

from training.losses import act_loss


class Trainer(nn.Module):
    def __init__(self, model, optimizer, cfg):
        super().__init__()
        self.model = model
        self.opt = optimizer
        self.cfg = cfg

    def _build_halting(self, outputs, h_states):
        adaptive = bool(getattr(self.cfg.halting, "adaptive", True))
        if self.cfg.halting.enabled and adaptive and self.model.halting is not None:
            return self.model.halting(h_states)

        if self.cfg.halting.enabled and not adaptive:
            n = len(outputs)
            p = 1.0 / max(n, 1)
            halting_probs = [o.new_full((o.size(0), 1), p) for o in outputs]
            expected_steps = None
            return halting_probs, expected_steps

        halting_probs = [o.new_zeros((o.size(0), 1)) for o in outputs]
        halting_probs[-1] = halting_probs[-1] + 1.0
        expected_steps = None
        return halting_probs, expected_steps

    def train_step(self, batch):
        x, y = batch
        self.model.train()
        outputs, h_states = self.model(x)

        halting_probs, expected_steps = self._build_halting(outputs, h_states)
        outputs_for_loss = outputs[: len(halting_probs)]

        loss = act_loss(
            outputs=outputs_for_loss,
            targets=y,
            halting_probs=halting_probs,
            ponder_cost=self.cfg.halting.ponder_cost,
            expected_steps=expected_steps,
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_clip = float(getattr(self.cfg.training, "grad_clip_norm", 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        self.opt.step()

        return float(loss.item())
