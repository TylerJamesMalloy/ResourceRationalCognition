"""
Module containing the utility prediction.
"""
import numpy as np

import torch
from torch import nn


# ALL encoders should be called Enccoder<Model>
def get_utility(utility_type):
    utility_type = utility_type.lower().capitalize()
    return eval("Utility{}".format(utility_type))


class UtilityMalloy(nn.Module):
    def __init__(self, latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        latent_dim : int
            Dimensionality of input to utility prediction module
            The input size is doubled as it takes in the sam

        Model Architecture 
        ------------
        - Model Input: 20 units (log variance and mean for 10 Gaussians)
        - 2 fully connected layers (each with 256 hidden values)
        - Model Output: 1 utility prediction (log variance and mean for a single contextual bandit stimuli utility prediction)
        

        References:
            [1] Modelling Human Information Processing Limitations in Learning Tasks with Reinforcement Learning
            T Malloy, CR Sims
            Proceedings of the 18th International Conference on Cognitive Modelling,
        """
        super(UtilityMalloy, self).__init__()

        # Layer parameters
        hidden_dim = 256
        self.latent_dim = latent_dim # model input 
        self.utility_out = 1 # number of utility predictions 

        # Fully connected layers
        self.lin1 = nn.Linear(self.latent_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Output layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.utility_out * 2)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.view((batch_size, -1))
        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.utility_out * 2).unbind(-1) 

        return mu, logvar

