import torch
from torch import nn
from torch.nn import functional as F


class SinCosPositionEncoding(nn.Module):

    def __init__(self, context_size, embed_size):
        super().__init__()
        # generate the positional encoding to select from
        self.encoding = nn.Embedding.from_pretrained(
                self.get_sin_cos_embeddings(context_size=context_size, embed_size=embed_size))

    @staticmethod
    def get_sin_cos_embeddings(context_size, embed_size):
        emb = torch.zeros(context_size, embed_size, requires_grad=False)
        pos = torch.arange(context_size, requires_grad=False)
        for i in range(embed_size//2):
            emb[:, 2 * i] = torch.sin(pos / 10000 ** (2 * i / embed_size))
            emb[:, 2 * i + 1] = torch.cos(pos / 10000 ** (2 * i / embed_size))
        return emb

    def forward(self, t):  # t is Tensor dims (T,)
        return self.encoding(t)


class AttentionHead(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(
        self,
        context_size: int,
        embed_size: int,
        head_size: int,
        dropout: float,
        decoder: True,
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
        self.dropout = nn.Dropout(p=dropout)

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
        # over the context_len dimension -> (B, context_len, context_len) with each
        # row summing to 1
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # dropout over the weights (regularization)
        # todo maybe experiment with this more.
        weights = self.dropout(weights)
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
        dropout: float,
        decoder: True,
    ):
        super().__init__()
        self.head_size = head_size  # H
        self.heads = nn.ModuleList(
            AttentionHead(
                context_size=context_size,
                embed_size=embed_size,
                head_size=head_size,
                dropout=dropout,
                decoder=decoder,
            )
            for _ in range(n_heads)
        )
        # this layer ensures that the output dim is always head_size.  Is that what I
        # want? maybe should be embed_size
        # seems like one rationale is that this projects back into the residual
        # pathway. To do with gradients.
        self.out_layer = nn.Linear(
            in_features=head_size * n_heads, out_features=embed_size
        )
        # dropout overs the projection -
        # todo experiment with this. Think about interpretability?
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, H * N)
        x = self.out_layer(x)  # (B, T, C)
        out = self.dropout(x)
        return out


class FeedforwardNet(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=embed_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=embed_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        context_size: int,
        embed_size: int,
        head_size: int,
        hidden_size: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attention_head = MultiHeadAttention(
            context_size=context_size,
            embed_size=embed_size,
            head_size=head_size,
            n_heads=n_heads,
            dropout=dropout,
            decoder=True,
        )
        self.attention_norm = nn.LayerNorm(embed_size)
        self.feedforward = FeedforwardNet(
            embed_size=embed_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.feedforward_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # this time we will add the residual connections and norm layers
        # x is (B, T)
        x = x + self.attention_head(self.attention_norm(x))
        out = x + self.feedforward(self.feedforward_norm(x))
        return out


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
