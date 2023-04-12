import os
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tokenizer import Tokenizer, IndexTokenizer

project_base_dir = os.path.dirname(os.path.abspath(__file__))


class ShakespeareDataset(Dataset):
    def __init__(self, file_location, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        with open(file_location, "r", encoding="utf8") as f:
            data = f.read()
        self.tokenizer.fit(data)
        self.encoded_data = self.tokenizer.encode(data)

    def __getitem__(self, index) -> Tensor:
        pass


# Basic data handling functionality - for the initial implementation
# to be rewritten using Dataset and Dataloader
# this provides a behaviour specification for the above.


class BasicShakespeareDataset:
    def __init__(
        self,
        filename,
        tokenizer: Tokenizer,
        context_size: int,
        batch_size: int,
        val_proportion: float,
        device="cpu",
        **kwargs,
    ):
        """

        :param filename:
        :param tokenizer:
        :param context_size: dimension of the context (i.e. length of an input in time
        dimension)
        :param batch_size: size of a batch
        :param val_proportion:
        :param device:
        """
        self.context_len = context_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        with open(filename, "r", encoding="utf8") as f:
            data = f.read()
        self.tokenizer.fit(data)
        encoded_data = torch.tensor(
            self.tokenizer.encode(data), dtype=torch.long, device=device
        )
        # very simple train val split here - TODO improve
        self.train_data = encoded_data[
            : round(len(encoded_data) * (1 - val_proportion))
        ]
        self.val_data = encoded_data[round(len(encoded_data) * val_proportion) + 1 :]

    def get_batch(self, split: str) -> Tuple[Tensor, Tensor]:
        """
        Return a batch of the dataset.
        :param split: train, val, test
        :return: Tuple[Tensor, Tensor] x, y dims [batch, context_len]
        """
        split_dataset_map = {"train": self.train_data, "val": self.val_data}
        if split not in split_dataset_map.keys():
            raise ValueError("Options are 'train' or 'val'. Try again!")
        # select some random indices for batch starting points
        indices = torch.randint(
            high=len(split_dataset_map[split]) - self.context_len,
            size=(self.batch_size,),
        )
        x = torch.stack(
            [split_dataset_map[split][idx : idx + self.context_len] for idx in indices]
        )
        y = torch.stack(
            [
                split_dataset_map[split][idx + 1 : idx + self.context_len + 1]
                for idx in indices
            ]
        )
        return x, y


if __name__ == "__main__":
    data_filename = os.path.join(project_base_dir, "data/input.txt")
    tokenizer = IndexTokenizer()
    context_len = 8
    batch_size = 4
    val_proportion = 0.1

    dataset = BasicShakespeareDataset(
        filename=data_filename,
        tokenizer=tokenizer,
        context_size=context_len,
        batch_size=batch_size,
        val_proportion=val_proportion,
    )
    x, y = dataset.get_batch(split="train")

    print(f"x shape: {x.shape}\n" f"x      : {x}")
    print(f"y shape: {y.shape}\n" f"y      : {y}")

    # visualize the context target matching.
    for b in range(batch_size):
        for t in range(context_len):
            context = x[b][: t + 1]
            target = y[b][t]
            print(f"Context: {context}, target: {target}")
