import logging
import os
from pathlib import Path
from typing import Dict

import torch
from omegaconf import DictConfig, OmegaConf

from chargpt.dataset import BasicShakespeareDataset, ShakespeareDataset
from chargpt.model import TransformerMultiBlockLanguageModel
from chargpt.tokenizer import IndexTokenizer

logger = logging.getLogger(__name__)

project_base_dir = os.path.dirname(os.path.abspath(__file__))
print(project_base_dir)


def evaluate(model, tokenizer, device, num_tokens):
    #### After sample #####
    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    model.eval()
    print(
        f"\n##### After #####\n"
        f"{tokenizer.decode(model.generate(inputs, tokens=num_tokens)[0])}"
        f"\n##### After #####"
    )
    #### After sample #####


def setup_evaluation(checkpoint_dir: Path, checkpoint: str):
    config: DictConfig = OmegaConf.load(checkpoint_dir / ".hydra" / "config.yaml")  # type: ignore
    # device = available_device() if config["device"] == "available" else config[
    #     "device"]
    device = torch.device("cpu")
    tok = IndexTokenizer()
    data_filename = os.path.join(
        project_base_dir,
        f"{config['data']['data_dir']}/" f"{config['data']['data_filename']}",
    )

    _ = ShakespeareDataset(
        filename=data_filename,
        tokenizer=tok,
        device=device,
        **config["shared"],
    )

    model = TransformerMultiBlockLanguageModel(
        vocab_size=tok.vocab_size,
        **config["shared"],
        **config["model"],
    )

    checkpoint_dict: Dict = torch.load(checkpoint_dir / "checkpoints" / checkpoint)

    model.load_state_dict(checkpoint_dict["model_state_dict"])
    model.to(device=device)

    return model, tok, device


if __name__ == "__main__":
    torch.manual_seed(45)

    # checkpoint_dir = Path(os.path.join(project_base_dir,
    #                                    "multirun/2023-08-21/16-51-53/0/"))
    # checkpoint = "final.pt"

    checkpoint_dir = Path(
        os.path.join(project_base_dir, "multirun/2023-08-21/16-51-53/1/")
    )
    checkpoint = "checkpoint_7000.pt"

    # num_tokens = 256
    # num_tokens = 512
    # num_tokens = 1024
    num_tokens = 2048
    # num_tokens = 4096

    model, tok, device = setup_evaluation(checkpoint_dir, checkpoint)
    evaluate(model=model, tokenizer=tok, device=device, num_tokens=num_tokens)
