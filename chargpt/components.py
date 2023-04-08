import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(
        self, context_size: int, embed_size: int, head_size: int, decoder: True
    ):
        super().__init__()
        self.head_size = head_size  # H
        self.decoder = decoder
        self.query = nn.Linear(embed_size, head_size, bias=False)  # (C, H)
        self.key = nn.Linear(embed_size, head_size, bias=False)  # (C, H)
        self.value = nn.Linear(embed_size, head_size, bias=False)  # (C, H)
        self.register_buffer(
            "mask", torch.tril(torch.ones(context_size, context_size))
        )  # (T, T)

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        weights = (
            q @ k.transpose(-2, -1) * self.head_size**-0.5
        )  # (B, T, H) @ (B, H, T) -> (B, T, T)
        if self.decoder:
            # NOTE: T x T section of mask for flexibility in input dims.
            weights = weights.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        # over the context_len dimension -> (B, context_len, context_len) with each row summing to 1
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        out = weights @ v  # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - embed_size (or head size of previous layer)
    # H: head dimension - head_size
    # N: num heads

    def __init__(
        self,
        context_size: int,
        embed_size: int,
        head_size: int,
        n_heads: int,
        decoder: True,
    ):
        super().__init__()
        self.head_size = head_size  # H
        self.heads = nn.ModuleList(
            AttentionHead(
                context_size=context_size,
                embed_size=embed_size,
                head_size=head_size,
                decoder=decoder,
            )
            for _ in range(n_heads)
        )
        # this layer ensures that the output dim is always head_size. Is that what I want? maybe should be embed_size
        self.out_layer = nn.Linear(
            in_features=head_size * n_heads, out_features=embed_size
        )

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, H * N)
        out = self.out_layer(out)
        return out


class FeedforwardNet(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=embed_size, out_features=embed_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(4, 8, 2)
    head = AttentionHead(context_size=8, embed_size=2, head_size=4, decoder=True)
    print(head(x))

    x_multi = torch.randn(4, 8, 2)
    multi_head = MultiHeadAttention(
        context_size=8, embed_size=2, head_size=4, n_heads=4, decoder=True
    )
    print(multi_head(x_multi))
