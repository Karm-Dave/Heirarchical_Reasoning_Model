# models/gm_hrm.py
import torch
import torch.nn as nn

from models.fusion import GatedFusion
from models.halting import ACTHalting
from models.hrm_modules import HighLevelModule, LowLevelModule
from models.memory import ExternalMemory


class GMHRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        d = cfg.model.hidden_dim
        in_dim = cfg.model.input_dim

        self.input_proj = nn.Linear(in_dim, d) if in_dim != d else nn.Identity()

        self.L = LowLevelModule(d)
        self.H = HighLevelModule(d)
        self.fusion = GatedFusion(d)

        self.max_segments = cfg.halting.max_segments
        self.memory_reset_each_forward = cfg.memory.reset_each_forward

        self.memory = (
            ExternalMemory(cfg.memory.slots, cfg.memory.slot_dim, d)
            if cfg.memory.enabled
            else None
        )

        self.halting = (
            ACTHalting(
                dim=d,
                max_segments=cfg.halting.max_segments,
                threshold=cfg.halting.threshold,
            )
            if cfg.halting.enabled
            else None
        )

        self.output = nn.Linear(d, cfg.model.output_dim)

    def forward(self, x):
        x_proj = self.input_proj(x)

        bsz = x_proj.shape[0]
        d = x_proj.shape[-1]
        z_l = x_proj.new_zeros((bsz, d))
        z_h = x_proj.new_zeros((bsz, d))

        if self.memory is not None and self.memory_reset_each_forward:
            self.memory.reset()

        h_states = []
        outputs = []

        for _ in range(self.max_segments):
            if self.memory is not None:
                z_l = z_l + self.memory.read(z_l)

            z_l = self.L(x_proj, z_l)
            mixed = self.fusion(z_l, z_h, x_proj)
            z_h = self.H(mixed, z_h)

            if self.memory is not None:
                self.memory.write(z_h)

            h_states.append(z_h)
            outputs.append(self.output(z_h))

        return outputs, h_states
