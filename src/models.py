"""Cells transcribed from the professor's appendix; no built-in recurrent layer."""
import torch
from torch import nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, variant='vanilla'):
        super().__init__()
        if variant not in ('vanilla', 'cifg', 'wmc'):
            raise ValueError(variant)
        self.variant, self.hidden_size = variant, hidden_size
        gates = 3 if variant == 'cifg' else 4
        self.x = nn.Linear(input_size, gates * hidden_size)
        self.h = nn.Linear(hidden_size, gates * hidden_size, bias=False)
        if variant == 'wmc':
            self.ci = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cf = nn.Linear(hidden_size, hidden_size, bias=False)
            self.co = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.ci = nn.Identity()
            self.cf = nn.Identity()
            self.co = nn.Identity()

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        return self.from_projected(self.x(x), state)

    def from_projected(self, projected: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = (projected + self.h(h)).chunk(3 if self.variant == 'cifg' else 4, -1)
        if self.variant == 'cifg':
            f, o, g = gates
            f = f.sigmoid()
            i = 1 - f
        else:
            i, f, o, g = gates
            if self.variant == 'wmc':
                i = i + self.ci(c).tanh()
                f = f + self.cf(c).tanh()
            i, f = i.sigmoid(), f.sigmoid()
        c = f * c + i * g.tanh()
        if self.variant == 'wmc':
            o = o + self.co(c).tanh()  # output gate reads CURRENT cell state
        return o.sigmoid() * c.tanh(), c


class Classifier(nn.Module):
    def __init__(self, vocab_size, variant, embedding_dim=64, hidden_size=48, dropout=0.2,
                 pooling='last', forget_bias=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell = LSTMCell(embedding_dim, hidden_size, variant)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 3)
        self.hidden_size = hidden_size
        if pooling not in ('last', 'mean'):
            raise ValueError(pooling)
        self.pooling = pooling
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            start = 0 if variant == 'cifg' else hidden_size
            self.cell.x.bias[start:start + hidden_size].fill_(forget_bias)

    def forward(self, tokens, lengths):
        embeddings = self.embedding(tokens)
        # Exactly the same affine input transform, computed for all timesteps at once.
        projected = self.cell.x(embeddings)
        h = embeddings.new_zeros(tokens.size(0), self.hidden_size)
        c = torch.zeros_like(h)
        pooled = torch.zeros_like(h)
        for t in range(tokens.size(1)):
            new_h, new_c = self.cell.from_projected(projected[:, t], (h, c))
            active = (t < lengths).unsqueeze(1)
            h, c = torch.where(active, new_h, h), torch.where(active, new_c, c)
            if self.pooling == 'mean':
                pooled = pooled + h * active
        if self.pooling == 'mean':
            h = pooled / lengths.clamp_min(1).unsqueeze(1)
        return self.output(self.dropout(h))
