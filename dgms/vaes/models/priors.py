import torch
import torch.distributions as td
import torch.nn as nn

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MixtureGaussianPrior(nn.Module):
    def __init__(self, M, num_components):
        """
        Define a Mixture of Gaussians prior distribution with zero mean and unit variance.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        num_components: [int]
           Number of mixture components in the Gaussian Mixture Model.
        """
        super(MixtureGaussianPrior, self).__init__()
        self.M = M
        self.num_components = num_components

        # Mixture component parameters (learnable parameters)
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))  # Mixing coefficients (unnormalized)
        self.means = nn.Parameter(torch.randn(num_components, M))  # Mean vectors for each component
        self.scales = nn.Parameter(torch.ones(num_components, M))  # Standard deviations

    def forward(self):
        """
        Return the Mixture of Gaussians prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # Define mixture components as Gaussian distributions
        component_distribution = td.Independent(
            td.Normal(loc=self.means, scale=self.scales), 1
        )

        # Define categorical mixing probabilities
        mixture_distribution = td.Categorical(logits=self.mixture_logits)

        # Construct MoG using MixtureSameFamily
        return td.MixtureSameFamily(mixture_distribution, component_distribution)
    


class SpatialGaussianPrior(nn.Module):
    def __init__(self, M, spatial_size=(7, 7)):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        spatial_size: [tuple] 
           The spatial dimensions of the latent representation.
        """
        super().__init__()
        self.M = M
        self.spatial_size = spatial_size  # (7, 7) for CNN-based encoding
        self.mean = nn.Parameter(torch.zeros((self.M, *self.spatial_size)), requires_grad=False)
        self.std = nn.Parameter(torch.ones((self.M, *self.spatial_size)), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 3)  # 3D latent space
