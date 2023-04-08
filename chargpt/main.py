import os

import torch
from tqdm import tqdm

from dataset import BasicShakespeareDataset
from model import (
    BigramLanguageModel,
    TransformerMultiBlockLanguageModel,
)
from tokenizer import IndexTokenizer

project_base_dir = os.path.dirname(os.path.abspath(__file__))


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


# TODO set up params better to use config for experimentation
def train_language_model(model, dataset, optimizer, eval_iters):
    loss = None
    losses = []
    for step in tqdm(range(iterations)):
        x, y = dataset.get_batch("train")

        if step % validate_interval == 0:
            checkpoint_losses = evaluate_val(
                model=model, dataset=dataset, eval_iters=eval_iters
            )
            losses.append(
                f"Step {step} Train loss: {checkpoint_losses['train']:.4f} | Val loss: {checkpoint_losses['val']:.4f}"
            )
            # print(f"Val loss at {step}: {avg_val_loss}")

        model.zero_grad(set_to_none=True)
        # ^ set to none is default True in 2.0 (should save mem but may cost in allocation?)
        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()
        optimizer.step()

    return model, loss.item(), losses


if __name__ == "__main__":
    torch.manual_seed(42)

    data_filename = os.path.join(project_base_dir, "data/input.txt")
    tok = IndexTokenizer()

    context_size = 128
    batch_size = 32
    embed_size = 256
    hidden_size = 4 * embed_size
    head_size = 64  # was 32 for single attention
    n_heads = 4
    n_blocks = 2

    dropout = 0.2

    val_proportion = 0.1
    validate_interval = 500
    eval_iters = 50
    iterations = 5000

    lr = 2e-4

    # device = "cpu"  # much faster on cpu until this size model. with these params cpu 8 mps 18
    device = available_device()
    print(device)

    dataset = BasicShakespeareDataset(
        filename=data_filename,
        tokenizer=tok,
        n_context=context_size,
        n_batch=batch_size,
        val_proportion=val_proportion,
        device=device,
    )

    model = TransformerMultiBlockLanguageModel(
        context_size=context_size,
        vocab_size=tok.vocab_size,
        embed_size=embed_size,
        head_size=head_size,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
    )
    model.to(device)

    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Before\n#####")
    model.eval()
    print(tok.decode(model.generate(inputs, max_new_tokens=200)[0]))
    print("#####\n")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    model.train()
    trained_model, final_loss, losses = train_language_model(
        model=model, dataset=dataset, optimizer=optimizer, eval_iters=eval_iters
    )

    print(f"Final Loss: {final_loss}")
    for item in losses:
        print(item)

    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\nAfter\n#####")
    trained_model.eval()
    print(tok.decode(trained_model.generate(inputs, max_new_tokens=200)[0]))
    print("#####\n")
