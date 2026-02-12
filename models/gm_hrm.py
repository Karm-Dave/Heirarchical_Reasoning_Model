from hrm_modules import LowLevelModule,HighLevelModule
from fusion import GatedFusion
from memory import ExternalMemory
from halting import ACTHalting
import torch
import torch.nn as nn

class GMHRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        d = cfg.model.hidden_dim
        self.L = LowLevelModule(d)
        self.H = HighLevelModule(d)
        self.fusion = GatedFusion(d)

        self.memory = ExternalMemory(
            cfg.memory.slots, d
        ) if cfg.memory.enabled else None

        self.halting = ACTHalting(d) if cfg.halting.enabled else None
        self.output = nn.Linear(d, cfg.model.output_dim)

    def forward(self, x):
        z_l = torch.zeros_like(x)
        z_h = torch.zeros_like(x)

        h_states = []
        outputs = []

        for segment in range(10):
            if self.memory:
                mem = self.memory.read(z_l)
                z_l = z_l + mem

            z_l = self.L(x, z_l)
            mixed = self.fusion(z_l, z_h, x)
            z_h = self.H(mixed, z_h)

            if self.memory:
                self.memory.write(z_h)

            h_states.append(z_h)
            outputs.append(self.output(z_h))

        return outputs, h_states
