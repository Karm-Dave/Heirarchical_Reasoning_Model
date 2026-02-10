class LowLevelModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.GRUCell(dim, dim)

    def forward(self, x, h):
        return self.rnn(x,h)
    
class HighLevelModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rnn = nn.GRUCell(dim, dim)

    def forward(self, z, h):
        return self.rnn(z, h)