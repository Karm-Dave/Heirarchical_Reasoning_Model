import torch
import torch.nn as nn
from training.losses import act_loss

class Trainer(nn.Module):
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.opt = optimizer
        self.cfg = cfg

    def train_step(self, batch):
        x, y = batch
        outputs, h_states = self.model(x)

        halting_probs = self.model.halting(h_states)
        loss = act_loss(outputs, y, halting_probs, self.cfg.halting.ponder_cost)

        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return loss.item()