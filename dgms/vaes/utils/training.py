from tqdm import tqdm

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.

    Returns:
    losses: [list]
        A list of loss values for tracking training progress.
    """
    model.train()
    losses = []

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    for epoch in range(epochs):
        epoch_losses = []  # Track losses for this epoch

        for batch_idx, (x, _) in enumerate(data_loader):  # Explicitly unpack images, ignore labels
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)  # Compute negative ELBO
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())  # Store loss after backprop

            # Update progress bar after every batch
            progress_bar.update(1)

        # Store the mean loss for this epoch
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)

        # Update progress bar with epoch-level stats
        progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", loss=f"{epoch_loss:.4f}")

    progress_bar.close()
    return losses