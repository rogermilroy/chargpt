from collections import defaultdict
import logging
import os

import hydra
import torch
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from chargpt.dataset import ShakespeareDataset, partition_dataset
from chargpt.hooks import Checkpoint, TextSample, TrainingMetric, ValidationMetric
from chargpt.model import (
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

    base_dataset = ShakespeareDataset(
        filename=data_filename,
        tokenizer=tok,
        device=device,
        **cfg["shared"],
    )

    train, test = partition_dataset(
        base_dataset, test_proportion=cfg["data"]["test_proportion"], **cfg["shared"]
    )

    train, val = partition_dataset(train, test_proportion=cfg["data"]["val_proportion"], **cfg["shared"])  # type: ignore

    train_dataloader = DataLoader(train, **cfg["dataloading"])
    val_dataloader = DataLoader(val, **cfg["dataloading"])
    test_dataloader = DataLoader(test, **cfg["dataloading"])

    sample_tokens = 200

    model = TransformerMultiBlockLanguageModel(
        vocab_size=tok.vocab_size,
        **cfg["shared"],
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

    # TODO - extract hooks setup into a helper function?
    post_hooks = list()
    metrics_dict = defaultdict(dict)
    samples = dict()
    if cfg["hooks"]["validate"]:
        post_hooks.append(
            ValidationMetric(
                metrics_dict=metrics_dict,
                dataloader=val_dataloader,
                interval=cfg["hooks"]["validate"]["interval"],
            )
        )
        post_hooks.append(
            TrainingMetric(
                metrics_dict=metrics_dict,
                interval=cfg["hooks"]["validate"]["interval"],
                dataloader=val_dataloader,
            )
        )
    if cfg["hooks"]["sample"]:
        post_hooks.append(
            TextSample(
                samples=samples,
                device=device,
                tokenizer=tok,
                **cfg["hooks"]["sample"],
                **cfg["shared"],
            )
        )
    if cfg["hooks"]["checkpoint"]:
        os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)
        post_hooks.append(Checkpoint(**cfg["hooks"]["checkpoint"]))

    model.train()
    trained_model, test_loss = train_language_model(
        epochs=cfg["run"]["epochs"],
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        post_hooks=post_hooks,
    )
    if cfg["save_final"]:
        torch.save(
            {
                "epoch": cfg["run"]["epochs"],
                "minibatch": len(
                    train_dataloader
                ),  # TODO think more about this - sub integer epochs...
                "model_state_dict": trained_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_loss": test_loss,
            },
            # TODO maybe change filename to match normal naming scheme?
            os.path.join(os.getcwd(), os.path.join("checkpoints", "final.pt")),
        )

    # TODO remove all this - when flushing losses directly in hooks

    ### process the metrics in the metrics dict. Output them
    metric_pretty_pre = {
        "training_loss": "Training Loss:",
        "validation_loss": "Validation Loss:",
    }
    metric_pretty_post = {
        "training_loss": " | ",
        "validation_loss": " | ",
    }
    # TODO fix cumulative tokens.
    for key, metrics_values in metrics_dict.items():
        epoch, minibatch = key
        output = []
        output.append(
            f"Epoch: {epoch} Minibatch: {minibatch} Cumulative Tokens: {minibatch * cfg['shared']['context_size'] * cfg['dataloading']['batch_size']} | "
        )
        for metric, value in metrics_values.items():
            output.append(
                f"{metric_pretty_pre[metric]} {value:.4f} {metric_pretty_post[metric]}"
            )
        logger.info("".join(output))

    # TODO simplify this more?
    for key, sample in samples.items():
        epoch, minibatch = key
        output = []
        output.append(
            f"Epoch: {epoch} Minibatch: {minibatch} Cumulative Tokens: {minibatch * cfg['shared']['context_size'] * cfg['dataloading']['batch_size']}\n"
        )
        output.append(f"### Sample ###\n{sample}\n### End Sample ###")
        logger.info("".join(output))

    #### After sample #####
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    trained_model.eval()
    logger.info(
        f"\n##### After #####\n"
        f"{tok.decode(trained_model.generate(inputs, tokens=200)[0])}"
        f"\n##### After #####"
    )
    #### After sample #####


if __name__ == "__main__":
    torch.manual_seed(42)
    run_training()
