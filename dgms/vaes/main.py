# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

# NOTE: Heavily modified and partitioned into separate files for better modularity and readability.
# David Miles-Skov 2025-02-07

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 

# Models
from models.vaes import (
    VAE,
    VAE_MoG,
)
from models.decoders import (
    BernoulliDecoder,
)
from models.encoders import (
    GaussianEncoder,
)
from models.priors import (
    GaussianPrior,
    MixtureGaussianPrior,
)
# utils
from utils.training import (
    train,
)
from utils.evaluating import (
    evaluate_elbo,
    visualize_latent_space_2d,
    visualize_latent_space_with_pca,
)

# class GaussianPrior(nn.Module):
#     def __init__(self, M):
#         """
#         Define a Gaussian prior distribution with zero mean and unit variance.

#                 Parameters:
#         M: [int] 
#            Dimension of the latent space.
#         """
#         super(GaussianPrior, self).__init__()
#         self.M = M
#         self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
#         self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

#     def forward(self):
#         """
#         Return the prior distribution.

#         Returns:
#         prior: [torch.distributions.Distribution]
#         """
#         return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


# class MixtureGaussianPrior(nn.Module):
#     def __init__(self, M, num_components):
#         """
#         Define a Mixture of Gaussians prior distribution with zero mean and unit variance.

#         Parameters:
#         M: [int] 
#            Dimension of the latent space.
#         num_components: [int]
#            Number of mixture components in the Gaussian Mixture Model.
#         """
#         super(MixtureGaussianPrior, self).__init__()
#         self.M = M
#         self.num_components = num_components

#         # Mixture component parameters (learnable parameters)
#         self.mixture_logits = nn.Parameter(torch.zeros(num_components))  # Mixing coefficients (unnormalized)
#         self.means = nn.Parameter(torch.randn(num_components, M))  # Mean vectors for each component
#         self.scales = nn.Parameter(torch.ones(num_components, M))  # Standard deviations

#     def forward(self):
#         """
#         Return the Mixture of Gaussians prior distribution.

#         Returns:
#         prior: [torch.distributions.Distribution]
#         """
#         # Define mixture components as Gaussian distributions
#         component_distribution = td.Independent(
#             td.Normal(loc=self.means, scale=self.scales), 1
#         )

#         # Define categorical mixing probabilities
#         mixture_distribution = td.Categorical(logits=self.mixture_logits)

#         # Construct MoG using MixtureSameFamily
#         return td.MixtureSameFamily(mixture_distribution, component_distribution)



# class GaussianEncoder(nn.Module):
#     def __init__(self, encoder_net):
#         """
#         Define a Gaussian encoder distribution based on a given encoder network.

#         Parameters:
#         encoder_net: [torch.nn.Module]             
#            The encoder network that takes as a tensor of dim `(batch_size,
#            feature_dim1, feature_dim2)` and output a tensor of dimension
#            `(batch_size, 2M)`, where M is the dimension of the latent space.
#         """
#         super(GaussianEncoder, self).__init__()
#         self.encoder_net = encoder_net

#     def forward(self, x):
#         """
#         Given a batch of data, return a Gaussian distribution over the latent space.

#         Parameters:
#         x: [torch.Tensor] 
#            A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
#         """
#         mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1) # NOTE: Output via the encoder (latent space representation)
#         return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) # NOTE: Return the Gaussian distribution corresponding to the latent space representation


# class BernoulliDecoder(nn.Module):
#     def __init__(self, decoder_net):
#         """
#         Define a Bernoulli decoder distribution based on a given decoder network.

#         Parameters: 
#         encoder_net: [torch.nn.Module]             
#            The decoder network that takes as a tensor of dim `(batch_size, M) as
#            input, where M is the dimension of the latent space, and outputs a
#            tensor of dimension (batch_size, feature_dim1, feature_dim2).
#         """
#         super(BernoulliDecoder, self).__init__()
#         self.decoder_net = decoder_net
#         self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

#     def forward(self, z):
#         """
#         Given a batch of latent variables, return a Bernoulli distribution over the data space.

#         Parameters:
#         z: [torch.Tensor] 
#            A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
#         """
#         logits = self.decoder_net(z)
#         return td.Independent(td.Bernoulli(logits=logits), 2) 
#         # used to convert a batch of independent distributions into a single multivariate distribution by treating
#         # multiple independent random variables as a joint distribution.


# class VAE(nn.Module):
#     """
#     Define a Variational Autoencoder (VAE) model.
#     """
#     def __init__(self, prior, decoder, encoder):
#         """
#         Parameters:
#         prior: [torch.nn.Module] 
#            The prior distribution over the latent space.
#         decoder: [torch.nn.Module]
#               The decoder distribution over the data space.
#         encoder: [torch.nn.Module]
#                 The encoder distribution over the latent space.
#         """
            
#         super(VAE, self).__init__()
#         self.prior = prior
#         self.decoder = decoder
#         self.encoder = encoder

#     def elbo(self, x):
#         """
#         Compute the ELBO for the given batch of data.

#         Parameters:
#         x: [torch.Tensor] 
#            A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
#            n_samples: [int]
#            Number of samples to use for the Monte Carlo estimate of the ELBO.
#         """
#         q = self.encoder(x) # NOTE: Get the Gaussian distribution object over the latent space.
#         z = q.rsample() # NOTE: Here is where the reparameterization trick is used. 
#         elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
#         return elbo

#     def sample(self, n_samples=1):
#         """
#         Sample from the model.
        
#         Parameters:
#         n_samples: [int]
#            Number of samples to generate.
#         """
#         z = self.prior().sample(torch.Size([n_samples]))
#         return self.decoder(z).sample()
    
#     def forward(self, x):
#         """
#         Compute the negative ELBO for the given batch of data.

#         Parameters:
#         x: [torch.Tensor] 
#            A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
#         """
#         return -self.elbo(x)
    
# class VAE_MoG(nn.Module):
#     """
#     Define a Variational Autoencoder (VAE) with a Mixture of Gaussians (MoG) prior.
#     """
#     def __init__(self, prior, decoder, encoder):
#         """
#         Parameters:
#         prior: [torch.nn.Module] 
#             The Mixture of Gaussians prior distribution over the latent space.
#         decoder: [torch.nn.Module]
#             The decoder distribution over the data space.
#         encoder: [torch.nn.Module]
#             The encoder distribution over the latent space.
#         """
#         super(VAE_MoG, self).__init__()
#         self.prior = prior  # Mixture of Gaussians prior
#         self.decoder = decoder
#         self.encoder = encoder



    
#     def elbo(self, x, num_samples=1):
#         """
#         Compute the ELBO for the given batch of data.

#         Parameters:
#         x: [torch.Tensor] 
#             A tensor of dimension `(batch_size, 28, 28)` representing input images.
#         num_samples: [int]
#             Number of Monte Carlo samples to use for KL divergence estimation.

#         Returns:
#         elbo: [torch.Tensor]
#             The estimated ELBO for the batch.
#         """
#         # 1. Approximate posterior q(z | x)
#         q = self.encoder(x)  # Gaussian posterior
#         z_samples = q.rsample(torch.Size([num_samples]))  # Monte Carlo samples from q(z|x)

#         # 2. Compute log likelihood term: log p(x | z)
#         log_px_given_z = self.decoder(z_samples).log_prob(x)  # Shape: (num_samples, batch_size, 28, 28)

#         # 3. Estimate KL divergence via Monte Carlo:
#         log_qz_given_x = q.log_prob(z_samples)  # log q(z | x), shape: (num_samples, batch_size)
#         log_pz = self.prior().log_prob(z_samples)  # log p(z), shape: (num_samples, batch_size)

#         kl_estimate = log_qz_given_x - log_pz  # KL term per sample, per batch

#         # 4. Compute ELBO:
#         elbo = log_px_given_z - kl_estimate  # Sum over image dimensions (28x28)

#         # 5. Average over Monte Carlo samples and batch
#         elbo = elbo.mean(dim=0)  # Average over Monte Carlo samples
#         elbo = elbo.mean()  # Average over batch
        
#         return elbo

    # def sample(self, n_samples=1):
    #     """
    #     Sample from the model.
        
    #     Parameters:
    #     n_samples: [int]
    #        Number of samples to generate.
    #     """
    #     z = self.prior().sample(torch.Size([n_samples]))
    #     return self.decoder(z).sample()
    
    # def forward(self, x):
    #     """
    #     Compute the negative ELBO for the given batch of data.

    #     Parameters:
    #     x: [torch.Tensor] 
    #        A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
    #     """
    #     return -self.elbo(x)



# def evaluate_elbo(model, data_loader, device):
#     """
#     Evaluate the ELBO on the test set.

#     Parameters:
#     model: [VAE] 
#         The trained VAE model.
#     data_loader: [torch.utils.data.DataLoader] 
#         The data loader for the test set.
#     device: [torch.device] 
#         Device to run the evaluation on.

#     Returns:
#     avg_elbo: [float] 
#         Average ELBO over the test set.
#     """
#     model.eval()
#     total_elbo = 0.0
#     num_samples = 0
#     num_batches = len(data_loader)

#     with torch.no_grad():
#         for x, _ in data_loader:  # x is the batch of images, _ ignores labels
#             # x: batch size x 28 x 28 (images are unflattened)
#             x = x.to(device)
#             elbo = model.elbo(x)  # Elbo is mean over batch!
#             total_elbo += elbo.sum().item()  # Sum over batch
#             num_samples += x.shape[0]  # Count total samples

#     # NOTE: total_elbo is sum of means over batch. to get mean over entire test set, divide by num_batches.
#     avg_elbo = total_elbo / num_batches 
#     print(f"Average ELBO on test set: {avg_elbo:.4f}")
#     return avg_elbo

# def train(model, optimizer, data_loader, epochs, device):
#     """
#     Train a VAE model.

#     Parameters:
#     model: [VAE]
#        The VAE model to train.
#     optimizer: [torch.optim.Optimizer]
#          The optimizer to use for training.
#     data_loader: [torch.utils.data.DataLoader]
#             The data loader to use for training.
#     epochs: [int]
#         Number of epochs to train for.
#     device: [torch.device]
#         The device to use for training.
#     """
#     model.train()

#     total_steps = len(data_loader)*epochs
#     progress_bar = tqdm(range(total_steps), desc="Training")
#     losses = []

#     for epoch in range(epochs):
#         data_iter = iter(data_loader)
#         for x, _ in data_iter:
#             x = x.to(device)
#             optimizer.zero_grad()
#             loss = model(x)
#             losses.append(loss.item())
#             loss.backward()
#             optimizer.step()

#             # Update progress bar
#             progress_bar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
#             progress_bar.update()
#     return losses

# def train(model, optimizer, data_loader, epochs, device):
#     """
#     Train a VAE model.

#     Parameters:
#     model: [VAE]
#        The VAE model to train.
#     optimizer: [torch.optim.Optimizer]
#          The optimizer to use for training.
#     data_loader: [torch.utils.data.DataLoader]
#             The data loader to use for training.
#     epochs: [int]
#         Number of epochs to train for.
#     device: [torch.device]
#         The device to use for training.

#     Returns:
#     losses: [list]
#         A list of loss values for tracking training progress.
#     """
#     model.train()
#     losses = []

#     total_steps = len(data_loader) * epochs
#     progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

#     for epoch in range(epochs):
#         epoch_losses = []  # Track losses for this epoch

#         for batch_idx, (x, _) in enumerate(data_loader):  # Explicitly unpack images, ignore labels
#             x = x.to(device)
#             optimizer.zero_grad()
#             loss = model(x)  # Compute negative ELBO
#             loss.backward()
#             optimizer.step()

#             epoch_losses.append(loss.item())  # Store loss after backprop

#             # Update progress bar after every batch
#             progress_bar.update(1)

#         # Store the mean loss for this epoch
#         epoch_loss = sum(epoch_losses) / len(epoch_losses)
#         losses.append(epoch_loss)

#         # Update progress bar with epoch-level stats
#         progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", loss=f"{epoch_loss:.4f}")

#     progress_bar.close()
#     return losses


# def visualize_latent_space_2d(model, data_loader, device, num_samples=1000):
#     """
#     Visualize samples from the approximate posterior (encoder output) and color them by class.

#     Simply plots the first two dimensions of the latent space.

#     Parameters:
#     model: [VAE] 
#         The trained VAE model.
#     data_loader: [torch.utils.data.DataLoader] 
#         The test set data loader.
#     device: [torch.device] 
#         The device to run computations on.
#     num_samples: [int] 
#         Number of samples to plot.
#     """
#     model.eval()
#     zs = []
#     labels = []

#     with torch.no_grad():
#         for x, y in data_loader: 
#             # x: batch_size, 28, 28
#             # y: batch_size,
#             x = x.to(device)
#             q = model.encoder(x)  # Encoder outputs the posterior q(z|x)
#             z = q.rsample()  # Sample from the posterior q(z|x) using reparameterization trick

#             zs.append(z.cpu())  # Store latent vectors
#             labels.append(y.cpu())  # Store corresponding labels
#             # len(zs) * x.shape[0] = num_batches * batch_size
#             if len(zs) * x.shape[0] >= num_samples:  # Stop after collecting enough samples
#                 break

#     # Stack collected latent variables and labels
#     zs = torch.cat(zs, dim=0).numpy()
#     labels = torch.cat(labels, dim=0).numpy()

#     # Plot the first two dimensions of `z`
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=zs[:, 0], y=zs[:, 1], hue=labels, palette="tab10", alpha=0.7)
#     plt.xlabel("Latent Dimension 1")
#     plt.ylabel("Latent Dimension 2")
#     plt.title("Samples from the Approximate Posterior (Latent Space)")
#     plt.legend(title="Digit Class")
#     plt.grid()
#     plt.savefig('dgms/vaes/latent_space.png')
#     plt.show()

# from sklearn.decomposition import PCA
# def visualize_latent_space_with_pca(model, data_loader, device, num_samples=1000):
#     """
#     Visualize samples from the approximate posterior (encoder output), using PCA for M > 2.

#     Parameters:
#     model: [VAE] 
#         The trained VAE model.
#     data_loader: [torch.utils.data.DataLoader] 
#         The test set data loader.
#     device: [torch.device] 
#         The device to run computations on.
#     num_samples: [int] 
#         Number of samples to plot.
#     """
#     model.eval()
#     zs = []
#     labels = []

#     with torch.no_grad():
#         for x, y in data_loader:
#             x = x.to(device)
#             q = model.encoder(x)  # Encoder outputs q(z|x)
#             z = q.rsample()  # Sample from q(z|x)

#             zs.append(z.cpu())  # Store latent vectors
#             labels.append(y.cpu())  # Store corresponding labels

#             if len(zs) * x.shape[0] >= num_samples:  # Stop after collecting enough samples
#                 break

#     # Stack collected latent variables and labels
#     zs = torch.cat(zs, dim=0).numpy()
#     labels = torch.cat(labels, dim=0).numpy()

#     # Apply PCA if latent dimension M > 2
#     if zs.shape[1] > 2:
#         print(f"Applying PCA to reduce latent space from {zs.shape[1]}D to 2D...")
#         pca = PCA(n_components=2)
#         zs = pca.fit_transform(zs)
#         print(f"Explained variance by first two components: {pca.explained_variance_ratio_.sum():.4f}")

#     # Plot the PCA-transformed latent variables
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=zs[:, 0], y=zs[:, 1], hue=labels, palette="tab10", alpha=0.7)
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.title("PCA Projection of Samples from the Approximate Posterior")
#     plt.legend(title="Digit Class")
#     plt.grid()
#     plt.savefig('dgms/vaes/latent_space_pca.png')
#     plt.show()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'all'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='gaussian_prior.pt', choices=['gaussian_prior.pt', 'MoG_prior.pt', 'gaussian_output.pt', 'continuous_bernoulli_output.pt'], help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'threshold' and create data loaders
    threshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Latent dimension
    M = args.latent_dim

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), # Input is 28x28 = 784 (flattened image)
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2), # Output is 2M (mean and std of the Gaussian distribution). M = latent dimension
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )


    
    # Match case to build model
    match args.model:
        
        case 'gaussian_prior.pt':
            decoder = BernoulliDecoder(decoder_net)
            encoder = GaussianEncoder(encoder_net)
            prior = GaussianPrior(M)
            model = VAE(prior, decoder, encoder).to(device)
        
        case 'MoG_prior.pt':
            decoder = BernoulliDecoder(decoder_net)
            encoder = GaussianEncoder(encoder_net)
            num_components = 5  # Number of Gaussian components in the mixture
            prior = MixtureGaussianPrior(M, num_components)
            model = VAE_MoG(prior, decoder, encoder).to(device)
        
        case 'gaussian_output':
            raise NotImplementedError("Gaussian output not implemented yet.")
        
        case 'continuous_bernoulli_output':
            raise NotImplementedError("Continuous Bernoulli output not implemented yet.")


    # Run model
    match args.mode:
        case 'train':
            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train model
            losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device)

            # Save model
            torch.save(model.state_dict(), args.model)
        
        case 'sample':
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), f'figs/{args.model}_samples.png')
        
        case 'eval':
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
            avg_elbo = evaluate_elbo(model, mnist_test_loader, args.device)
            visualize_latent_space_2d(model, mnist_test_loader, args.device)
            visualize_latent_space_with_pca(model, mnist_test_loader, args.device)
        
        case 'all':

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Train model
            losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device)
            # Save model
            torch.save(model.state_dict(), args.model)
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample(64)).cpu() 
                save_image(samples.view(64, 1, 28, 28), f'figs/{model.name}_samples.png')
            # Evaluate model
            avg_elbo = evaluate_elbo(model, mnist_test_loader, args.device)
            # Visualize latent space
            visualize_latent_space_2d(model, mnist_test_loader, args.device)
            visualize_latent_space_with_pca(model, mnist_test_loader, args.device)



        

        

    # # Choose mode to run
    # if args.mode == 'train':
    #     # Define optimizer
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #     # Train model
    #     losses = train(model, optimizer, mnist_train_loader, args.epochs, args.device)


    #     # Save model
    #     torch.save(model.state_dict(), args.model)

    # elif args.mode == 'sample':
    #     model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

    #     # Generate samples
    #     model.eval()
    #     with torch.no_grad():
    #         samples = (model.sample(64)).cpu() 
    #         save_image(samples.view(64, 1, 28, 28), args.samples)

    # elif args.mode == "eval":
    #     model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    #     avg_elbo = evaluate_elbo(model, mnist_test_loader, args.device)

    # elif args.mode == "visualise_2d":
    #     model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    #     visualize_latent_space_2d(model, mnist_test_loader, args.device)
    # elif args.mode == "PCA":
    #     model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    #     visualize_latent_space_with_pca(model, mnist_test_loader, args.device)



# Training and evaluating model with original simple gaussian prior
# Training: uv run dgms/vaes/vae_bernoulli.py train --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model dgms/vaes/model.pt
# Sampling: uv run dgms/vaes/vae_bernoulli.py sample --device cpu --latent-dim 10 --model dgms/vaes/model.pt --samples dgms/vaes/samples.png 


# Training MOG prior model: uv run dgms/vaes/vae_bernoulli.py train --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model dgms/vaes/model_mog.pt 
# Sampling MOG prior model: uv run dgms/vaes/vae_bernoulli.py sample --device cpu --latent-dim 10 --model dgms/vaes/model_mog.pt --samples dgms/vaes/samples_mog.png
# evaluation: uv run dgms/vaes/vae_bernoulli.py eval --device cpu --latent-dim 10 --model dgms/vaes/model.pt
# visualise_2d: uv run dgms/vaes/vae_bernoulli.py visualise --device cpu --latent-dim 10 --model dgms/vaes/model.pt
# PCA: uv run dgms/vaes/vae_bernoulli.py PCA --device cpu --latent-dim 10 --model dgms/vaes/model.pt