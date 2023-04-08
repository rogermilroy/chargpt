import torch
from torch import nn
from torch.nn import functional as F

from components import AttentionHead


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # vocab_size x vocab_size table (direct probs for each char based on previous char) like Q table.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=vocab_size
        )

    def forward(self, x):
        # x is (B, T)
        return self.token_embedding(x)  # (B, T, C)

    def loss(self, logits, targets):
        b, t, c = logits.shape
        logits = logits.view((b*t, c))
        targets = targets.view(b*t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(probs, num_samples=1)  # select one from probs (B, 1, 1)

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
        self.attention_head = AttentionHead(context_size=context_size,
                                            embed_size=embed_size,
                                            head_size=self.head_size,
                                            decoder=True)
        self.output_layer = nn.Linear(in_features=self.head_size, out_features=vocab_size)

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
        logits = logits.view((b*t, c))
        targets = targets.view(b*t)
        return F.cross_entropy(logits, targets)

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            # left trim x to be last n_context tokens
            x_trim = x[:, -self.context_size:]

            logits = self(x_trim)  # logits (B, T, C) C is output options

            logits = logits[:, -1, :]  # select last time step from logits (B, 1, C)
            probs = F.softmax(logits, dim=-1)  # logits to probs
            x_next = torch.multinomial(probs, num_samples=1)  # select one from probs (B, 1, 1)

            x = torch.cat((x, x_next), dim=1)  # (B, T + 1, 1)
        return x
