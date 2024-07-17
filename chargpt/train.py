import math
from typing import Optional, List
from torch.utils.data import DataLoader

from tqdm import tqdm

from chargpt.hooks import Hook, validation_loss


# TODO set up params better to use config for experimentation
def train_language_model(
    epochs: float,
    model,
    optimizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    post_hooks: Optional[List[Hook]] = None,
):
    loss = None
    if post_hooks is None:
        post_hooks = list()

    # do some processing to allow sub epoch runs (ie. half a dataset = 0.5)
    minibatch_limit = None
    if 0 < epochs < 1:
        minibatch_limit = int(len(train_dataloader) * epochs)
        # TODO add printout for minibatch limit if it's set here.
    else:
        minibatch_limit = math.inf

    for epoch in tqdm(range(math.ceil(epochs)), desc="Epochs"):
        for i, minibatch in tqdm(
            enumerate(train_dataloader),
            desc="Minibatches",
            total=(
                len(train_dataloader)
                if minibatch_limit == math.inf
                else minibatch_limit
            ),
        ):
            if i > minibatch_limit:
                break
            x, y = minibatch

            model.zero_grad(set_to_none=True)
            # ^ set to none is default True in 2.0 (should save mem but may cost in allocation?)
            logits = model(x)
            loss = model.loss(logits, y)
            loss.backward()
            optimizer.step()

            for hook in post_hooks:
                # TODO improve this to be less restrictive - ie. passing in less stuff..
                hook(
                    epoch=epoch,
                    minibatch=i,
                    model=model,
                    optimizer=optimizer,
                    logits=logits,
                    loss=loss,
                )
    # test loss is the mean loss over the whole test dataset
    test_loss = validation_loss(model=model, dataloader=test_dataloader)

    return (
        model,
        test_loss,
    )
