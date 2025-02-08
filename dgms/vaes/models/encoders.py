import torch
import torch.distributions as td
import torch.nn as nn

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        print(x.shape)
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1) # NOTE: Output via the encoder (latent space representation)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1) # NOTE: Return the Gaussian distribution corresponding to the latent space representation

class GaussianEncoderCNN(GaussianEncoder):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super().__init__(encoder_net)
    
    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
        A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        if x.dim() == 3:  
            x = x.unsqueeze(1)  # Convert (batch_size, 28, 28) → (batch_size, 1, 28, 28)


        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)  
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)