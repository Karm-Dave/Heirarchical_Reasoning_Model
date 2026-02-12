import torch
import torch.nn as nn


class GatedFusion(nn.Module):


    def __init__(self, dim):
        super().__init__()

        self.dim = dim

       
        self.gate = nn.Linear(3 * dim, dim)

    def forward(self, z_l, z_h, x):

        concat = torch.cat([z_l, z_h, x], dim=-1)  

        g = torch.sigmoid(self.gate(concat))       

        mixed = g * z_h + (1 - g) * z_l

        return mixed
