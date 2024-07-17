from abc import ABC
import logging
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from chargpt.model import TorchLanguageModel

logger = logging.getLogger(__name__)


class Hook(ABC):
    def __call__(self, epoch, minibatch, model, logits, loss, **kwargs):
        ...


"""
Idea:
We keep a map of {
    epoch,minibatch: {metric: value, ...},

}
Then we iterate over it at the end.
"""


@torch.no_grad()
def validation_loss(model: TorchLanguageModel, dataloader: DataLoader) -> torch.Tensor:
    model.eval()

    losses = torch.zeros(len(dataloader))
    # TODO fix the train, val
    for i, minibatch in enumerate(dataloader):
        x, y = minibatch
        logits = model(x)
        loss = model.loss(logits=logits, targets=y)
        losses[i] = loss.item()
    out = losses.mean()

    model.train()
    return out


# TODO generify this - calculate logits in one step and then run metrics over the logits in another step.


class ValidationMetric(Hook):
    def __init__(
        self,
        metrics_dict: Dict,
        dataloader: DataLoader,
        interval: int,
    ) -> None:
        super().__init__()
        self.metrics_dict = metrics_dict
        self.dataloader: DataLoader = dataloader
        self.interval = interval

    def __call__(self, epoch, minibatch, model, **kwargs):
        if minibatch % self.interval == 0:
            checkpoint_losses = validation_loss(model=model, dataloader=self.dataloader)
            self.metrics_dict[(epoch, minibatch)]["validation_loss"] = checkpoint_losses


class TrainingMetric(Hook):
    def __init__(
        self,
        metrics_dict: Dict,
        dataloader: DataLoader,
        interval: int,
    ) -> None:
        super().__init__()
        # interval in minibatches
        self.metrics_dict = metrics_dict
        self.interval = interval
        self.losses = torch.zeros(self.interval)

    def __call__(self, epoch, minibatch, model, loss, logits, **kwargs):
        self.losses[minibatch % self.interval] = loss.item()
        if minibatch % self.interval == 0:
            self.metrics_dict[(epoch, minibatch)]["training_loss"] = self.losses.mean()
            # reset losses for the next interval
            self.losses = torch.zeros(self.interval)


class TextSample(Hook):
    def __init__(
        self,
        samples: Dict,
        interval: int,
        tokens: int,
        device,
        tokenizer,
        **kwargs,
    ):
        self.samples = samples
        self.interval = interval
        self.device = device
        self.tokenizer = tokenizer
        self.tokens = tokens

    def __call__(self, epoch, minibatch, model, **kwargs):
        if minibatch % self.interval == 0:
            inputs = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            model.eval()
            sample = f"{self.tokenizer.decode(model.generate(inputs, tokens=self.tokens)[0])}"
            model.train()
            self.samples[(epoch, minibatch)] = sample


class Checkpoint(Hook):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, epoch, minibatch, model, optimizer, loss, **kwargs):
        if minibatch % self.interval == 0:
            # create the checkpoint name - might want it to
            checkpoint_fname = os.path.join(
                os.getcwd(),
                os.path.join("checkpoints", f"checkpoint_{epoch}_{minibatch}.pt"),
            )
            logger.debug("Saving checkpoint")
            torch.save(
                {
                    "epoch": epoch,
                    "minibatch": minibatch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_fname,
            )
