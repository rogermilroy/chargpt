import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):

    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(self, context_size: int, embed_size: int, head_size: int, decoder: True):
        super().__init__()
        self.head_size = head_size  # H
        self.decoder = decoder
        self.query = nn.Linear(embed_size, head_size, bias=False)  # (C, H)
        self.key = nn.Linear(embed_size, head_size, bias=False)  # (C, H)
        self.value = nn.Linear(embed_size, head_size, bias=False)  # (C, H)
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))  # (T, T)

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, H) @ (B, H, T) -> (B, T, T)
        if self.decoder:
            # NOTE: T x T section of mask for flexibility in input dims.
            weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        # over the context_len dimension -> (B, context_len, context_len) with each row summing to 1
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        out = weights @ v  # (B, T, H)
        return out


if __name__ == '__main__':
    torch.manual_seed(42)

    x = torch.randn(4, 8, 2)
    head = AttentionHead(context_size=8,
                         embed_size=2,
                         head_size=4,
                         decoder=True
                         )
    print(head(x))
