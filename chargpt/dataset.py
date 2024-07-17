import os
from typing import Sized, Tuple, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader

from chargpt.tokenizer import Tokenizer, IndexTokenizer

project_base_dir = os.path.dirname(os.path.abspath(__file__))


class SizedDataset(Dataset, Sized):
    ...


class SizedSubset(Subset, Sized):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:  # type: ignore
        super().__init__(dataset, indices)
        self._size = len(indices)

    def __len__(self):
        return self._size


def partition_dataset(
    dataset: SizedDataset | SizedSubset,
    test_proportion: float,
    context_size: int,
) -> Tuple[SizedSubset, SizedSubset]:
    train = SizedSubset(
        dataset,
        range(round(len(dataset) * (1 - test_proportion)) - context_size),
    )
    test = SizedSubset(
        dataset, range(round(len(dataset) * (1 - test_proportion)) + 1, len(dataset))
    )

    return train, test


class ShakespeareDataset(SizedDataset):
    def __init__(
        self,
        filename,
        tokenizer: Tokenizer,
        context_size: int,
        device: str | torch.device = "cpu",
    ):
        self.tokenizer = tokenizer
        with open(filename, "r", encoding="utf8") as f:
            data = f.read()
        self.tokenizer.fit(data)
        encoded_data = torch.tensor(
            self.tokenizer.encode(data), dtype=torch.long, device="cpu"
        )
        # here stack sections of context size TODO find more efficient way to do this.
        self.x = torch.stack(
            [
                encoded_data[idx : idx + context_size]
                for idx in range(len(encoded_data) - context_size)
            ]
        ).to(device=device)

        self.y = torch.stack(
            [
                encoded_data[idx + 1 : idx + context_size + 1]
                for idx in range(len(encoded_data) - context_size)
            ]
        ).to(device=device)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]


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

    base_dataset = ShakespeareDataset(
        filename=data_filename,
        tokenizer=tokenizer,
        context_size=context_len,
    )

    # subsets for train/test
    train, test = partition_dataset(
        base_dataset,
        test_proportion=val_proportion,
        context_size=context_len,
    )

    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

    dataset = BasicShakespeareDataset(
        filename=data_filename,
        tokenizer=tokenizer,
        context_size=context_len,
        batch_size=batch_size,
        val_proportion=val_proportion,
    )
    x, y = dataset.get_batch(split="train")

    x_dl, y_dl = next(iter(train_dataloader))

    print(f"x shape: {x.shape}\n" f"x      : {x}")
    print(f"y shape: {y.shape}\n" f"y      : {y}")

    # visualize the context target matching.
    for b in range(batch_size):
        for t in range(context_len):
            context = x[b][: t + 1]
            target = y[b][t]
            print(f"Context: {context}, target: {target}")
