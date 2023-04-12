from tqdm import tqdm


# TODO set up params better to use config for experimentation
def train_language_model(
    model, dataset, optimizer, iterations, validate_interval, eval_iters, eval_fn
):
    loss = None
    losses = []
    for step in tqdm(range(iterations)):
        x, y = dataset.get_batch("train")

        if step % validate_interval == 0:
            checkpoint_losses = eval_fn(
                model=model, dataset=dataset, eval_iters=eval_iters
            )
            losses.append(
                f"Step {step} Train loss: {checkpoint_losses['train']:.4f} "
                f"| Val loss: {checkpoint_losses['val']:.4f}"
            )
            # print(f"Val loss at {step}: {avg_val_loss}")

        model.zero_grad(set_to_none=True)
        # ^ set to none is default True in 2.0 (should save mem but may cost in allocation?)
        logits = model(x)
        loss = model.loss(logits, y)
        loss.backward()
        optimizer.step()

    return model, loss.item(), losses
