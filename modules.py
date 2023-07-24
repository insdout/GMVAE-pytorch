import torch
from torch import nn

"""
====================================================================================
Used repositories:
link:
https://github.com/sghalebikesabi/gmm-vae-clustering-pytorch

link:
https://github.com/RuiShu/vae-clustering/blob/master/gmvae.py

link:
https://github.com/insdout/MDS-Thesis-RULPrediction/blob/main/models/tshae_models.py
====================================================================================
"""


class Qy_x(nn.Module):
    """Conditional distribution q(y|x) represented by a neural network.

    Args:
        encoder (nn.Module): The encoder module used to process the input data.
        enc_out_dim (int): The output dimension of the encoder module.
        k (int): Number of components in the Gaussian mixture prior.

    Attributes:
        h1 (nn.Module): The encoder module used to process the input data.
        qy_logit (nn.Linear): Linear layer for predicting the logit of q(y|x).
        qy (nn.Softmax): Softmax activation function for q(y|x).

    """
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.h1 = encoder
        self.qy_logit = nn.Linear(enc_out_dim, k)
        self.qy = nn.Softmax(dim=1)

    def forward(self, x):
        """Perform the forward pass for q(y|x).

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            tuple: A tuple containing the logit and softmax outputs of q(y|x).
        """
        h1 = self.h1(x)
        qy_logit = self.qy_logit(h1)
        qy = self.qy(qy_logit)
        return qy_logit, qy


class Qz_xy(nn.Module):
    """Conditional distribution q(z|x, y) represented by a neural network.

    Args:
        k (int): Number of components in the Gaussian mixture prior.
        encoder (nn.Module): The encoder module used to process the input data.
        enc_out_dim (int): The output dimension of the encoder module.
        hidden_size (int): Number of units in the hidden layer.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        h1 (nn.Module): The encoder module used to process the input data.
        h2 (nn.Sequential): The hidden layers of the neural network.
        z_mean (nn.Linear): Linear layer for predicting the mean of q(z|x, y).
        zlogvar (nn.Linear): Linear layer for predicting
            the log variance of q(z|x, y).

    """
    def __init__(self, k, encoder, enc_out_dim, hidden_size, latent_dim):
        super(Qz_xy, self).__init__()
        self.h1 = encoder
        self.h2 = nn.Sequential(
            nn.Linear(enc_out_dim + k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_size, latent_dim)
        self.zlogvar = nn.Linear(hidden_size, latent_dim)

    def gaussian_sample(self, z_mean, z_logvar):
        z_std = torch.sqrt(torch.exp(z_logvar))

        eps = torch.randn_like(z_std)
        z = z_mean + eps*z_std

        return z

    def forward(self, x, y):
        """Perform the forward pass for q(z|x, y).

        Args:
            x (torch.Tensor): Input data tensor.
            y (torch.Tensor): One-hot encoded tensor representing
                the class labels.

        Returns:
            tuple: A tuple containing the latent variables, mean,
                and log variance of q(z|x, y).
        """
        h1 = self.h1(x)
        xy = torch.cat((h1, y), dim=1)
        h2 = self.h2(xy)
        # q(z|x, y)
        z_mean = self.z_mean(h2)
        zlogvar = self.zlogvar(h2)
        z = self.gaussian_sample(z_mean, zlogvar)
        return z, z_mean, zlogvar


class Px_z(nn.Module):
    """Conditional distribution p(x|z) represented by a neural network.

    Args:
        decoder (nn.Module): The decoder module used to reconstruct the data.
        k (int): Number of components in the Gaussian mixture prior.

    Attributes:
        decoder (nn.Module): The decoder module used to reconstruct the data.
        decoder_hidden (int): Number of units in the hidden layer
            of the decoder.
        latent_dim (int): Dimensionality of the latent space.
        z_mean (nn.Linear): Linear layer for predicting the mean of p(z|y).
        zlogvar (nn.Linear): Linear layer for predicting the log variance
            of p(z|y).

    """
    def __init__(self, decoder, k):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.decoder_hidden = self.decoder.hidden_size
        self.latent_dim = self.decoder.latent_dim
        self.z_mean = nn.Linear(k, self.latent_dim)
        self.zlogvar = nn.Linear(k, self.latent_dim)

    def forward(self, z, y):
        """Perform the forward pass for p(x|z) and p(z|y).

        Args:
            z (torch.Tensor): Latent variable tensor.
            y (torch.Tensor): One-hot encoded tensor representing
                the class labels.

        Returns:
            tuple: A tuple containing the prior mean, prior log variance,
                and reconstructed data.
        """
        # p(z|y)
        z_mean = self.z_mean(y)
        zlogvar = self.zlogvar(y)

        # p(x|z)
        x_hat = self.decoder(z)
        return z_mean, zlogvar, x_hat


class EncoderFC(nn.Module):
    """Fully connected encoder module.

    Args:
        input_size (int): Dimensionality of the input data.
        hidden_size (int): Number of units in the hidden layer.
        dropout (float): Dropout probability.

    Attributes:
        input_size (int): Dimensionality of the input data.
        hidden_size (int): Number of units in the hidden layer.
        p (float): Dropout probability.
        enc_block (nn.Sequential): Sequential neural network layers
            for the encoder.

    """
    def __init__(self, input_size, hidden_size, dropout):
        super(EncoderFC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.p = dropout

        self.enc_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.p)
        )

    def forward(self, x):
        """Perform the forward pass for the encoder module.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Encoded tensor after passing through
                the encoder layers.
        """
        h = self.enc_block(x)
        return h


class DecoderFC(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, return_probs=True):
        """Fully connected decoder module.

        Args:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool, optional): Whether to apply a sigmoid
                activation for output probabilities. Defaults to True.

        Attributes:
            input_size (int): Dimensionality of the input data.
            hidden_size (int): Number of units in the hidden layer.
            latent_dim (int): Dimensionality of the latent space.
            return_probs (bool): Whether to apply a sigmoid activation
                for output probabilities.
            dec_block (nn.Sequential): Sequential neural network layers
                for the decoder.
            sigmoid (nn.Sigmoid): Sigmoid activation function.

        """

        super(DecoderFC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.return_probs = return_probs

        self.dec_block = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        """Perform the forward pass for the decoder module.

        Args:
            z (torch.Tensor): Latent variable tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor after passing
                through the decoder layers.
        """
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out
