import torch
import torch.distributions as td
import torch.nn as nn

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2) 
        # used to convert a batch of independent distributions into a single multivariate distribution by treating
        # multiple independent random variables as a joint distribution.


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net, latent_dim, learn_std=False):
        """
        Define a Gaussian decoder distribution based on a given decoder network.
        NOTE: Could not be bothered to implement static std. 

        Parameters:
        decoder_net: [torch.nn.Module] 
           The decoder network that takes a tensor of dim `(batch_size, M)`
           as input, where M is the latent space dimension, and outputs 
           a tensor of dimension `(batch_size, 28, 28)`.
        latent_dim: [int]
            The dimensionality of the latent space.
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
            A tensor of dimension `(batch_size, M)`, where M is the latent space dimension.
        """
        out = self.decoder_net(z)  # Shape: (batch_size, 2, 28, 28)

        mean = out[:, 0, :, :]  # First channel is mean (μ)
        log_var = out[:, 1, :, :]  # Second channel is log-variance (log σ²)

        std = torch.exp(0.5 * log_var)  # Convert log-variance to std-dev (σ = exp(0.5 * log σ²))

        return td.Independent(td.Normal(loc=mean, scale=std), 2)  # Multivariate Gaussian
