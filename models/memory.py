# models/memory.py
import torch
import torch.nn as nn


class ExternalMemory(nn.Module):
    def __init__(self, slots: int, slot_dim: int, query_dim: int):
        super().__init__()
        self.slots = slots
        self.slot_dim = slot_dim
        self.query_dim = query_dim

        self.register_buffer("memory", torch.zeros(slots, slot_dim))

        self.key = nn.Linear(query_dim, slot_dim)
        self.erase = nn.Linear(query_dim, slot_dim)
        self.add = nn.Linear(query_dim, slot_dim)

    def reset(self):
        self.memory.zero_()

    def read(self, query: torch.Tensor) -> torch.Tensor:
        q = self.key(query)  # (B, slot_dim)
        mem = self.memory.detach().clone()
        attn = torch.softmax(q @ mem.T, dim=-1)  # (B, slots)
        return attn @ mem  # (B, slot_dim)

    def write(self, state: torch.Tensor):
        k = self.key(state)  # (B, slot_dim)
        attn = torch.softmax(k @ self.memory.T, dim=-1)  # (B, slots)

        erase = torch.sigmoid(self.erase(state))  # (B, slot_dim)
        add = torch.tanh(self.add(state))  # (B, slot_dim)

        attn_mean = attn.mean(dim=0, keepdim=True).T  # (slots, 1)
        erase_mean = erase.mean(dim=0, keepdim=True)  # (1, slot_dim)
        add_mean = add.mean(dim=0, keepdim=True)  # (1, slot_dim)

        new_memory = self.memory * (1.0 - attn_mean * erase_mean) + (attn_mean * add_mean)
        with torch.no_grad():
            self.memory.copy_(new_memory)
