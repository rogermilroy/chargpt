import os

import torch
from tqdm import tqdm

from dataset import BasicShakespeareDataset
from model import BigramLanguageModel
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
def evaluate_val(model: BigramLanguageModel, dataset: BasicShakespeareDataset, eval_iters):
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
def train():

    data_filename = os.path.join(project_base_dir, "data/input.txt")
    tokenizer = IndexTokenizer()
    context_len = 4
    batch_size = 32
    val_proportion = 0.1
    validate_interval = 100
    device = 'cpu'  # much faster on cpu until much larger model I think.
    print(device)

    dataset = BasicShakespeareDataset(
        filename=data_filename,
        tokenizer=tokenizer,
        context_len=context_len,
        batch_size=batch_size,
        val_proportion=val_proportion,
        device=device,
    )

    model = BigramLanguageModel(tokenizer.vocab_size).to(device)

    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Before\n#####")
    print(tokenizer.decode(model.generate(inputs, max_new_tokens=100)[0]))
    print("#####\n")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)

    loss = None
    losses = []
    for step in tqdm(range(1000)):
        x, y = dataset.get_batch("train")

        if step % validate_interval == 0:
            checkpoint_losses = evaluate_val(model=model, dataset=dataset, eval_iters=20)
            losses.append(f"Step {step} Train loss: {checkpoint_losses['train']:.4f} | Val loss: {checkpoint_losses['val']:.4f}")
            # print(f"Val loss at {step}: {avg_val_loss}")

        model.zero_grad(set_to_none=True)
        # ^ set to none is default True in 2.0 (should save mem but may cost in allocation?)
        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()
        optimizer.step()
    print(f'Final Loss: {loss.item()}')
    for item in losses:
        print(item)

    inputs = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\nAfter\n#####")
    print(tokenizer.decode(model.generate(inputs, max_new_tokens=100)[0]))
    print("#####\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    train()
