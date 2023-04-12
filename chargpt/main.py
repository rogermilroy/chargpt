import logging
import os

import hydra
import torch
from omegaconf import DictConfig

from dataset import BasicShakespeareDataset
from model import (
    BigramLanguageModel,
    TransformerMultiBlockLanguageModel,
)
from tokenizer import IndexTokenizer
from train import train_language_model

project_base_dir = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger(__name__)


def available_device() -> str:
    if torch.has_mps:
        return "mps"
    elif torch.has_cuda:
        return "cuda"
    else:
        return "cpu"


@torch.no_grad()
def evaluate_val(
    model: BigramLanguageModel, dataset: BasicShakespeareDataset, eval_iters
):
    model.eval()

    out = dict()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = dataset.get_batch(split)
            logits = model(x)
            loss = model.loss(logits=logits, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_training(cfg: DictConfig):
    logger.info(cfg)

    data_filename = os.path.join(
        project_base_dir, f"{cfg['data']['data_dir']}/{cfg['data']['data_filename']}"
    )
    tok = IndexTokenizer()

    device = available_device() if cfg["device"] == "available" else cfg["device"]
    print(device)

    dataset = BasicShakespeareDataset(
        filename=data_filename,
        tokenizer=tok,
        device=device,
        context_size=cfg["context_size"],
        **cfg["data"],
    )

    model = TransformerMultiBlockLanguageModel(
        vocab_size=tok.vocab_size,
        context_size=cfg["context_size"],
        **cfg["model"],
    )
    model.to(device)

    #### Before sample #####
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Before\n#####")
    model.eval()
    print(tok.decode(model.generate(inputs, max_new_tokens=200)[0]))
    print("#####\n")
    #### Before sample #####

    optimizer = torch.optim.AdamW(params=model.parameters(), **cfg["optimizer"])

    model.train()
    trained_model, final_loss, losses = train_language_model(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        eval_fn=evaluate_val,
        **cfg["run"],
    )
    torch.save(trained_model, "./final.pt")

    print(f"Final Loss: {final_loss}")
    for item in losses:
        print(item)

    #### After sample #####
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\nAfter\n#####")
    trained_model.eval()
    print(tok.decode(trained_model.generate(inputs, max_new_tokens=200)[0]))
    print("#####\n")
    #### After sample #####


if __name__ == "__main__":
    run_training()
