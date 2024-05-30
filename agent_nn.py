import torch
from torch import nn
import numpy as np

class AgentNN(nn.Module):
    """
    define neural network architecture
    """
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Linear layers
        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        if freeze:
            self._freeze()
        
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def _get_conv_out(self, shape):
        """
        compute the output size after passing an input through 
        the convolutional layers. 
        """
        out = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(out.size()))
    
    def _freeze(self):  
        """
        iterates through all parameters of the network and 
        effectively freezes them.
        """      
        for param in self.network.parameters():
            param.requires_grad = False
    