import torch
import torch.nn as nn

class ExternalMemory(nn.Module):
    def __init__(self, slots, dim):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.memory = nn.Parameter(torch.zeros(slots, dim))

        self.key = nn.Linear(dim, dim)
        self.erase = nn.Linear(dim, dim)
        self.add = nn.Linear(dim, dim)

    def read(self, query):
        attn = torch.softmax(
            torch.matmul(self.key(query), self.momery.T), dim = -1
        )
        return torch.matmul(attn, self.memory)

    def write(self, state):
        attn = torch.softmax(
            torch.matmul(self.key(state), self.memory.T), dim = -1
        )
        earse = torch.sigmoid(self.erase(state))
        add = torch.tanh(self.add(state))

        self.memory.data = self.memory.data * (1 - attn.unsqueeze(-1) * earse.unsqueeze(1))
        self.memory.data = self.memory.data + attn.unsqueeze(-1) * add.unsqueeze(1)