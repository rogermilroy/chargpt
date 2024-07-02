import torch
from torch import dropout, nn
from torch.nn import functional as F

from chargpt.components import (
    AttentionHead,
    MultiHeadAttention,
    FeedforwardNet,
    TransformerBlock,
    SinCosPositionEncoding,
)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # vocab_size x vocab_size table (direct probs for each char based on
        # previous char) like Q table.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, x):
        # x is (B, T)
        return self.token_embedding(x)  # (B, T, C)

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x


class AttentionLanguageModel(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(self, context_size, vocab_size, embed_size, head_size):
        """

        :param context_size: dimension of the context (ie. length of an input string)
        :param vocab_size: dimension of the vocabulary
        :param embed_size: dimension of the token and position embeddings
        :param head_size: dimension of the attention head
        """
        super().__init__()
        self.context_size = context_size
        self.head_size = head_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_size, embed_size)
        self.attention_head = AttentionHead(
            context_size=context_size,
            embed_size=embed_size,
            head_size=self.head_size,
            decoder=True,
        )
        self.output_layer = nn.Linear(
            in_features=self.head_size, out_features=vocab_size
        )

    def forward(self, x):
        # x is (B, T)
        _, t = x.shape
        x = self.token_embedding(x)  # (B, T, C)
        pos = self.position_embedding(torch.arange(t, device=x.device))  # (T, C)
        x = x + pos
        x = self.attention_head(x)
        out = self.output_layer(x)
        return out

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size :]

            logits = self(x_trim)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x


class MultiHeadAttentionLanguageModel(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(self, context_size, vocab_size, embed_size, head_size, n_heads):
        """

        :param context_size: dimension of the context (ie. length of an input string)
        :param vocab_size: dimension of the vocabulary
        :param embed_size: dimension of the token and position embeddings
        :param head_size: dimension of the attention head - usually computed from
            embed_size and n_heads - embed_size // n_heads. Keeping separate for
            experimentation.
        :param n_heads: number of attention heads
        """
        super().__init__()
        self.context_size = context_size
        self.head_size = head_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_size, embed_size)
        self.attention_head = MultiHeadAttention(
            context_size=context_size,
            embed_size=embed_size,
            head_size=self.head_size,
            n_heads=n_heads,
            decoder=True,
        )
        # self.feedforward = FeedforwardNet(embed_size=embed_size)
        self.output_layer = nn.Linear(in_features=embed_size, out_features=vocab_size)

    def forward(self, x):
        # x is (B, T)
        _, t = x.shape
        x = self.token_embedding(x)  # (B, T, C)
        pos = self.position_embedding(torch.arange(t, device=x.device))  # (T, C)
        x = x + pos
        x = self.attention_head(x)
        out = self.output_layer(x)
        return out

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size :]

            logits = self(x_trim)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x


class MultiHeadAttentionFFLanguageModel(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(
        self, context_size, vocab_size, embed_size, head_size, n_heads, hidden_size
    ):
        """

        :param context_size: dimension of the context (ie. length of an input string)
        :param vocab_size: dimension of the vocabulary
        :param embed_size: dimension of the token and position embeddings
        :param head_size: dimension of the attention head - usually computed from
            embed_size and n_heads - embed_size // n_heads. Keeping separate for
            experimentation.
        :param n_heads: number of attention heads
        """
        super().__init__()
        self.context_size = context_size
        self.head_size = head_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_size, embed_size)
        self.attention_head = MultiHeadAttention(
            context_size=context_size,
            embed_size=embed_size,
            head_size=self.head_size,
            n_heads=n_heads,
            decoder=True,
        )
        self.feedforward = FeedforwardNet(
            embed_size=embed_size, hidden_size=hidden_size, dropout=0.0
        )
        self.output_layer = nn.Linear(in_features=embed_size, out_features=vocab_size)

    def forward(self, x):
        # x is (B, T)
        _, t = x.shape
        x = self.token_embedding(x)  # (B, T, C)
        pos = self.position_embedding(torch.arange(t, device=x.device))  # (T, C)
        x = x + pos
        x = self.attention_head(x)
        x = self.feedforward(x)
        out = self.output_layer(x)
        return out

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size :]

            logits = self(x_trim)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x


class TransformerMultiBlockLanguageModel(nn.Module):
    # B: batch size
    # T: time dimension - context_len
    # C: channels - n_embed (or head size of previous layer)
    # H: head dimension - head_size

    def __init__(
        self,
        context_size,
        vocab_size,
        embed_size,
        head_size,
        hidden_size,
        n_heads,
        n_blocks,
        dropout,
        pos_embedding="sin_cos",
    ):
        """
        Creates a Multi Block Transformer model. That is a stack of MultiHeadAttention
        Blocks parameterised accordingly.
        :param context_size: dimension of the context (ie. length of an input string)
        :param vocab_size: dimension of the vocabulary
        :param embed_size: dimension of the token and position embeddings
        :param head_size: dimension of the attention head - usually computed from
            embed_size and n_heads - embed_size // n_heads.
            Keeping separate for experimentation.
        :param hidden_size: dimension of the feedforward networks in the Attention
            blocks
        :param n_heads: number of attention heads per block
        :param n_blocks: number of blocks - analogous to depth of the Transformer
        :param dropout: proportion of dropout applied (to Attention heads and
            feedforward nets)
        :param pos_embedding: the position embedding technique used
            {sin_cos | learned} default sin_cos.
        """
        super().__init__()
        self.context_size = context_size
        self.head_size = head_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        if pos_embedding == "learned":
            self.position_embedding = nn.Embedding(context_size, embed_size)
        elif pos_embedding == "sin_cos":
            self.position_embedding = SinCosPositionEncoding(
                context_size=context_size, embed_size=embed_size
            )
        else:
            raise ValueError(
                f"pos_embedding must be one of 'learned' or 'sin_cos' "
                f"found: {pos_embedding}"
            )
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_size=context_size,
                    embed_size=embed_size,
                    head_size=head_size,
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    dropout=dropout,
                )
            ]
            * n_blocks
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        # TODO tie these weights to token embedding
        self.output_layer = nn.Linear(in_features=embed_size, out_features=vocab_size)

    def forward(self, x):
        # this time we will add the residual connections and norm layers
        # x is (B, T)
        _, t = x.shape
        x = self.token_embedding(x)  # (B, T, C)
        pos = self.position_embedding(torch.arange(t, device=x.device))  # (T, C)
        x = x + pos
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        out = self.output_layer(x)
        return out

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b * t, c))
        targets = targets.view(b * t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size :]

            logits = self(x_trim)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(
                probs, num_samples=1
            )  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x
