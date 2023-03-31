"""
Tokenizer - initially just the simplest possible char level tokenizer.
"""
from abc import ABC, abstractmethod
from typing import List
import torch

# additional options - sub word tokenizers
# sentencepiece - Google
# tiktoken - bpe tokenizer - OpenAI BytePairEncoding

# NOTES: trade off is between vocab size and context length
# (in general shorter context lengths are better for sequence
# models as they often struggle to propagate information along the length,
# vocab size increases number of weight)
# char is the extreme, sequence length gets very long but vocab is "small",
# word level is usually the other extreme?
from torch import Tensor


class Tokenizer(ABC):
    @abstractmethod
    def fit(self, data):
        ...

    @abstractmethod
    def encode(self, string: str) -> Tensor:
        ...

    @abstractmethod
    def decode(self, tokens: Tensor) -> str:
        ...


class IndexTokenizer(Tokenizer):
    def __init__(self):
        self.str_to_tok = None
        self.tok_to_str = None

    def fit(self, data: str | List[str]):
        """
        Some procedure to fit to a data set. TODO finish
        :param data:
        :return: IndexTokenizer
        """
        if isinstance(data, List):
            data = "".join(data)
        chars = sorted(list(set(data)))
        self.str_to_tok = {ch: i for i, ch in enumerate(chars)}
        self.tok_to_str = {i: ch for i, ch in enumerate(chars)}
        print(f"Vocab size: {len(chars)}")
        return self

    def encode(self, string: str) -> Tensor:
        if self.str_to_tok is None:
            raise AttributeError(
                "IndexTokenizer not initialized - you need to fit to some data!"
            )
        return torch.tensor([self.str_to_tok[ch] for ch in string], dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        if self.tok_to_str is None:
            raise AttributeError(
                "IndexTokenizer not initialized - you need to fit to some data!"
            )
        # note this assumes 1d only for now.
        return "".join([self.tok_to_str[tok] for tok in tokens.tolist()])


if __name__ == "__main__":
    """Simple inline testing for sanity check"""
    init = "Some testing string here!"
    tokenizer = IndexTokenizer().fit(init)
    print(tokenizer.encode(init))
    print(tokenizer.decode(tokenizer.encode(init)))
