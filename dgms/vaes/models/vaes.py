import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F

"""
Collection of VAE models used for week 1.
"""


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, name="basic_vae"):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.name=name

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x) # NOTE: Get the Gaussian distribution object over the latent space.
        z = q.rsample() # NOTE: Here is where the reparameterization trick is used. 
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


class VAE_MoG(VAE):
    """
    VAE with altered ELBO implementation to handle mixture of Gaussians prior.
    """
    def __init__(self, prior, decoder, encoder, name="vae_gaussian_output"):
        """
        Parameters:
        prior: [torch.nn.Module] 
            The prior distribution over the latent space.
        decoder: [torch.nn.Module]
            The Gaussian decoder distribution over the data space.
        encoder: [torch.nn.Module]
            The encoder distribution over the latent space.
        """
        super().__init__(prior, decoder, encoder, name)
    
    def elbo(self, x, num_samples=1):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, 28, 28)` representing input images.
        num_samples: [int]
            Number of Monte Carlo samples to use for KL divergence estimation.

        Returns:
        elbo: [torch.Tensor]
            The estimated ELBO for the batch.
        """
        # 1. Approximate posterior q(z | x)
        q = self.encoder(x)  # Gaussian posterior
        z_samples = q.rsample(torch.Size([num_samples]))  # Monte Carlo samples from q(z|x)

        # 2. Compute log likelihood term: log p(x | z)
        log_px_given_z = self.decoder(z_samples).log_prob(x)  # Shape: (num_samples, batch_size, 28, 28)

        # 3. Estimate KL divergence via Monte Carlo:
        log_qz_given_x = q.log_prob(z_samples)  # log q(z | x), shape: (num_samples, batch_size)
        log_pz = self.prior().log_prob(z_samples)  # log p(z), shape: (num_samples, batch_size)

        kl_estimate = log_qz_given_x - log_pz  # KL term per sample, per batch

        # 4. Compute ELBO:
        elbo = log_px_given_z - kl_estimate  # Sum over image dimensions (28x28)

        # 5. Average over Monte Carlo samples and batch
        elbo = elbo.mean(dim=0)  # Average over Monte Carlo samples
        elbo = elbo.mean()  # Average over batch
        
        return elbo

    

class VAE_gaussian_output(VAE):
    """
    Variational Autoencoder (VAE) with a Gaussian output distribution.
    """

    def __init__(self, prior, decoder, encoder, name="vae_gaussian_output"):
        """
        Parameters:
        prior: [torch.nn.Module] 
            The prior distribution over the latent space.
        decoder: [torch.nn.Module]
            The Gaussian decoder distribution over the data space.
        encoder: [torch.nn.Module]
            The encoder distribution over the latent space.
        """
        super().__init__(prior, decoder, encoder, name)


class VAE_CNN(VAE):
    def __init__(self, prior, decoder, encoder, name="vae_cnn"):
        """
        Parameters:
        prior: [torch.nn.Module] 
            The prior distribution over the latent space.
        decoder: [torch.nn.Module]
            The Gaussian decoder distribution over the data space.
        encoder: [torch.nn.Module]
            The encoder distribution over the latent space.
        """
        super().__init__(prior, decoder, encoder, name)
    
    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
        A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`

        Returns:
        elbo: [torch.Tensor]
        A scalar tensor representing the average ELBO for the batch.
        """
        q = self.encoder(x)  # Gaussian posterior q(z | x)
        z = q.rsample()  # Sample using reparameterization trick

        log_px_given_z = self.decoder(z).log_prob(x)  # Log-likelihood per image
        kl_div = td.kl_divergence(q, self.prior())  # KL divergence per image

        elbo_per_image = log_px_given_z - kl_div  # Compute ELBO per image
        elbo = elbo_per_image.mean()  # Take mean over batch

        return elbo
