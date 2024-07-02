import logging
import os

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from chargpt.dataset import BasicShakespeareDataset
from chargpt.hooks import Validate, Checkpoint
from chargpt.model import (
    BigramLanguageModel,
    TransformerMultiBlockLanguageModel,
)
from chargpt.tokenizer import IndexTokenizer
from chargpt.train import train_language_model

project_base_dir = os.path.dirname(os.path.abspath(__file__))


logger = logging.getLogger(__name__)


def available_device() -> str:
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.backends.cuda.is_built():
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
    out_dir = HydraConfig.get().runtime.output_dir
    logger.debug(
        f"CWD: {os.getcwd()}, project root: {project_base_dir}, output_dir: "
        f"{out_dir}"
    )

    data_filename = os.path.join(
        project_base_dir, f"{cfg['data']['data_dir']}/{cfg['data']['data_filename']}"
    )
    tok = IndexTokenizer()

    device = available_device() if cfg["device"] == "available" else cfg["device"]
    logger.info(f"Device: {device}")

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

    # try loading weights
    if cfg["run"].get("resume") is not None:
        logger.info(f"resuming from : {cfg['run'].get('resume')}")
        checkpoint = torch.load(
            os.path.join(project_base_dir, cfg["run"]["resume"]["path"])
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    else:
        logger.info("Starting from scratch")
        optimizer = torch.optim.AdamW(params=model.parameters(), **cfg["optimizer"])

    #### Before sample #####
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    logger.info(
        f"\n##### Before #####\n"
        f"{tok.decode(model.generate(inputs, max_new_tokens=200)[0])}"
        f"\n##### Before #####"
    )
    #### Before sample #####

    post_hooks = list()
    if cfg["run"]["validate"]:
        post_hooks.append(
            Validate(
                eval_fn=evaluate_val,
                **cfg["run"]["validate"],
            )
        )
    if cfg["run"]["checkpoint"]:
        os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
        post_hooks.append(Checkpoint(**cfg["run"]["checkpoint"]))

    model.train()
    trained_model, final_loss, losses = train_language_model(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        post_hooks=post_hooks,
        iterations=cfg["run"]["iterations"],
    )
    if cfg["save_final"]:
        torch.save(
            {
                "step": cfg["run"]["iterations"],
                "model_state_dict": trained_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": final_loss,
            },
            os.path.join(os.getcwd(), os.path.join("checkpoints", "final.pt")),
        )

    for item in losses:
        logger.info(item)
    logger.info(f"Final Loss: {final_loss}")

    #### After sample #####
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    trained_model.eval()
    logger.info(
        f"\n##### After #####\n"
        f"{tok.decode(trained_model.generate(inputs, max_new_tokens=200)[0])}"
        f"\n##### After #####"
    )
    #### After sample #####


if __name__ == "__main__":
    torch.manual_seed(42)
    run_training()
