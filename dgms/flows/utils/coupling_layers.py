import torch
import torch.nn as nn

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Define a coupling layer.

        Parameters:
        scale_net: [torch.nn.Module]
            The scaling network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        translation_net: [torch.nn.Module]
            The translation network that takes as input a tensor of dimension `(batch_size, feature_dim)` and outputs a tensor of dimension `(batch_size, feature_dim)`.
        mask: [torch.Tensor]
            A binary mask of dimension `(feature_dim,)` that determines which features (where the mask is zero) are transformed by the scaling and translation networks.
        """
        super(MaskedCouplingLayer, self).__init__()
        # Set the scaling and translation networks as attributes
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, x):
        """
        Implements the forward transformation of a masked coupling layer.

        T(z) = b * z + (1 - b) * (z * exp(s(b*z) + t(b*z)))

        where b is the mask, s and t are the scaling and translation networks, respectively.

        also returns the log-determinant of the Jacobian matrix of the forward transformation.

        log abs(det J_T(z)) = sum_{i=1}^D (1-b_i) s_i (b * z)

        Transform a batch of data through the coupling layer (from the base to data).

        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations of dimension `(batch_size, feature_dim)`.
        """
        # z = x
        # log_det_J = torch.zeros(x.shape[0])
        b = self.mask
        s = self.scale_net(b * x)
        t = self.translation_net(b * x)
        
        z = b * x + (1 - b) * (x * torch.exp(s) + t)


        sum_log_det_J = torch.sum((1 - b) * s, dim=1)

        return z, sum_log_det_J
    
    def inverse(self, z):
        """

        Implements the inverse of the forward transformation of a masked coupling layer.

        T^{-1}(x) = b * x + (1 - b) * (x - t(b*x)) * exp(-s(b*x))

        where b is the mask, s and t are the scaling and translation networks, respectively.


        Transform a batch of data through the coupling layer (from data to the base).

        Parameters:
        z: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        x: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        # z = x
        # log_det_J = torch.zeros(x.shape[0])

        b = self.mask
        s = self.scale_net(b * z)
        t = self.translation_net(b * z)

        x = b * z + (1 - b) * ((z - t) * torch.exp(-s))
        sum_log_det_J = -torch.sum((1 - b) * s, dim=1)

        return x, sum_log_det_J