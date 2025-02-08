import torch
import torch.distributions as td
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def evaluate_elbo(model, data_loader, device):
    """
    Evaluate the ELBO on the test set.

    Parameters:
    model: [VAE] 
        The trained VAE model.
    data_loader: [torch.utils.data.DataLoader] 
        The data loader for the test set.
    device: [torch.device] 
        Device to run the evaluation on.

    Returns:
    avg_elbo: [float] 
        Average ELBO over the test set.
    """
    model.eval()
    total_elbo = 0.0
    num_samples = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for x, _ in data_loader:  # x is the batch of images, _ ignores labels
            # x: batch size x 28 x 28 (images are unflattened)
            x = x.to(device)
            elbo = model.elbo(x)  # Elbo is mean over batch!
            total_elbo += elbo.sum().item()  # Sum over batch
            num_samples += x.shape[0]  # Count total samples

    # NOTE: total_elbo is sum of means over batches. to get mean over entire test set, divide by num_batches.
    avg_elbo = total_elbo / num_batches 
    print(f"Average ELBO on test set: {avg_elbo:.4f}")
    # Write value to file
    with open(f'data/elbo/{model.name}_elbo.txt', 'w') as f:
        f.write(f"Average ELBO on test set: {avg_elbo:.4f}")
    return avg_elbo


def visualize_latent_space_2d(model, data_loader, device, num_samples=1000):
    """
    Visualize samples from the approximate posterior (encoder output) and color them by class.

    Simply plots the first two dimensions of the latent space.

    Parameters:
    model: [VAE] 
        The trained VAE model.
    data_loader: [torch.utils.data.DataLoader] 
        The test set data loader.
    device: [torch.device] 
        The device to run computations on.
    num_samples: [int] 
        Number of samples to plot.
    """
    model.eval()
    zs = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader: 
            # x: batch_size, 28, 28
            # y: batch_size,
            x = x.to(device)
            q = model.encoder(x)  # Encoder outputs the posterior q(z|x)
            z = q.rsample()  # Sample from the posterior q(z|x) using reparameterization trick

            zs.append(z.cpu())  # Store latent vectors
            labels.append(y.cpu())  # Store corresponding labels
            # len(zs) * x.shape[0] = num_batches * batch_size
            if len(zs) * x.shape[0] >= num_samples:  # Stop after collecting enough samples
                break

    # Stack collected latent variables and labels
    zs = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Plot the first two dimensions of `z`
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=zs[:, 0], y=zs[:, 1], hue=labels, palette="tab10", alpha=0.7)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Samples from the Approximate Posterior (Latent Space)")
    plt.legend(title="Digit Class")
    plt.grid()
    plt.savefig(f'figs/{model.name}_latent_space.png')
    plt.show()


def visualize_latent_space_2d(model, data_loader, device, num_samples=1000):
    """
    Visualize samples from the approximate posterior (encoder output) and color them by class.

    Simply plots the first two dimensions of the latent space.

    Parameters:
    model: [VAE] 
        The trained VAE model.
    data_loader: [torch.utils.data.DataLoader] 
        The test set data loader.
    device: [torch.device] 
        The device to run computations on.
    num_samples: [int] 
        Number of samples to plot.
    """
    model.eval()
    zs = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader: 
            # x: batch_size, 28, 28
            # y: batch_size,
            x = x.to(device)
            q = model.encoder(x)  # Encoder outputs the posterior q(z|x)
            z = q.rsample()  # Sample from the posterior q(z|x) using reparameterization trick

            zs.append(z.cpu())  # Store latent vectors
            labels.append(y.cpu())  # Store corresponding labels
            # len(zs) * x.shape[0] = num_batches * batch_size
            if len(zs) * x.shape[0] >= num_samples:  # Stop after collecting enough samples
                break

    # Stack collected latent variables and labels
    zs = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Plot the first two dimensions of `z`
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=zs[:, 0], y=zs[:, 1], hue=labels, palette="tab10", alpha=0.7)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Samples from the Approximate Posterior (Latent Space)")
    plt.legend(title="Digit Class")
    plt.grid()
    plt.savefig(f'figs/{model.name}_latent_space.png')
    plt.show()



def visualize_latent_space_with_pca(model, data_loader, device, num_samples=1000):
    """
    Visualize samples from the approximate posterior (encoder output), using PCA for M > 2.

    Parameters:
    model: [VAE] 
        The trained VAE model.
    data_loader: [torch.utils.data.DataLoader] 
        The test set data loader.
    device: [torch.device] 
        The device to run computations on.
    num_samples: [int] 
        Number of samples to plot.
    """
    model.eval()
    zs = []
    labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x)  # Encoder outputs q(z|x)
            z = q.rsample()  # Sample from q(z|x)

            zs.append(z.cpu())  # Store latent vectors
            labels.append(y.cpu())  # Store corresponding labels

            if len(zs) * x.shape[0] >= num_samples:  # Stop after collecting enough samples
                break

    # Stack collected latent variables and labels
    zs = torch.cat(zs, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Apply PCA if latent dimension M > 2
    if zs.shape[1] > 2:
        print(f"Applying PCA to reduce latent space from {zs.shape[1]}D to 2D...")
        pca = PCA(n_components=2)
        zs = pca.fit_transform(zs)
        print(f"Explained variance by first two components: {pca.explained_variance_ratio_.sum():.4f}")

    # Plot the PCA-transformed latent variables
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=zs[:, 0], y=zs[:, 1], hue=labels, palette="tab10", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection of Samples from the Approximate Posterior")
    plt.legend(title="Digit Class")
    plt.grid()
    plt.savefig(f'figs/{model.name}_latent_space_pca.png')
    plt.show()