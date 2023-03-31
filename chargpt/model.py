from torch import nn


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
