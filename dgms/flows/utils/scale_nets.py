import torch
import torch.nn as nn

class BasicScaleNet(nn.Module):
    def __init__(self, D, num_hidden):
        """
        A simple scale network that takes an input of dimension D and returns a scale factor 
        of the same dimension.

        Parameters:
        D: [int] 
            The input and output dimension.
        num_hidden: [int] 
            The number of hidden units in the network.
        """
        super(BasicScaleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, D)
        )

    def forward(self, x):
        """
        Forward pass of the scale network.

        Parameters:
        x: [torch.Tensor] 
            Input tensor of shape (batch_size, D).

        Returns:
        torch.Tensor: 
            Output tensor of shape (batch_size, D) representing the scale factors.
        """
        return self.network(x)
    

class DeepScaleNet(nn.Module):
    def __init__(self, D, num_hidden, num_layers):
        """
        Deeper scaling network used for a flow-based model.
        
        Parameters:
        D: [int] 
            The input and output dimension.
        num_hidden: [int]
            The number of hidden units in each layer.
        num_layers: [int]
            The number of hidden layers in the network.
        """

        super(DeepScaleNet, self).__init__()
        self.D = D
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.network = self.__build_net__(D, num_hidden, num_layers)

    # Private method to build the scale network
    def __build_net__(self, D, num_hidden, num_layers):
        """
        Simple function to build the network.

        Parameters:
        D: [int] 
            The input and output dimension.
        num_hidden: [int]
            The number of hidden units in each layer.
        num_layers: [int]
            The number of hidden layers in the network.

        Returns:
        nn.Sequential: 
            The deep scale network.
        """
        layers = [nn.Linear(D, num_hidden), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hidden, D))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the deep scale network.

        Parameters:
        x: [torch.Tensor] 
            Input tensor of shape (batch_size, D).

        Returns:
        torch.Tensor: 
            Output tensor of shape (batch_size, D) representing the scale factors.
        """
        return self.network(x)