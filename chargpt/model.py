import torch
from torch import nn
from torch.nn import functional as F


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


