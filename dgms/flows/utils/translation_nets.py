import torch
import torch.nn as nn

class BasicTranslationNet(nn.Module):
    def __init__(self, D, num_hidden):
        """
        A simple translation network that takes an input of dimension D and returns a translation 
        factor of the same dimension.

        Parameters:
        D: [int] 
            The input and output dimension.
        num_hidden: [int] 
            The number of hidden units in the network.
        """
        super(BasicTranslationNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(D, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, D)
        )

    def forward(self, x):
        """
        Forward pass of the translation network.

        Parameters:
        x: [torch.Tensor] 
            Input tensor of shape (batch_size, D).

        Returns:
        torch.Tensor: 
            Output tensor of shape (batch_size, D) representing the translation factors.
        """
        return self.network(x)
    

class DeepTranslationNet(nn.Module):
    def __init__(self, D, num_hidden, num_layers):
        """
        Deeper translation network used for a flow-based model.
        
        Parameters:
        D: [int] 
            The input and output dimension.
        num_hidden: [int]
            The number of hidden units in each layer.
        num_layers: [int]
            The number of hidden layers in the network.
        """

        super(DeepTranslationNet, self).__init__()
        self.D = D
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.network = self.__build_net__(D, num_hidden, num_layers)

    # Private method to build the translation network
    def __build_net__(self, D, num_hidden, num_layers):
        """
        Build the translation network.
        
        Parameters:
        D: [int] 
            The input and output dimension.
        num_hidden: [int]
            The number of hidden units in each layer.
        num_layers: [int]
            The number of hidden layers in the network.
        """
        layers = []
        layers.append(nn.Linear(D, num_hidden))
        layers.append(nn.ReLU())
        for i in range(num_layers - 2):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_hidden, D))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the translation network.
        
        Parameters:
        x: [torch.Tensor] 
            Input tensor of shape (batch_size, D).
        
        Returns:
        torch.Tensor: 
            Output tensor of shape (batch_size, D) representing the translation factors.
        """
        return self.network(x)