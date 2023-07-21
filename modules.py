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
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.h1 = encoder
        self.qy_logit = nn.Linear(enc_out_dim, k)
        self.qy = nn.Softmax(dim=1)

    def forward(self, x):
        h1 = self.h1(x)
        qy_logit = self.qy_logit(h1)
        qy = self.qy(qy_logit)
        return qy_logit, qy


class Qz_xy(nn.Module):
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
        h1 = self.h1(x)
        xy = torch.cat((h1, y), dim=1)
        h2 = self.h2(xy)
        # q(z|x, y)
        z_mean = self.z_mean(h2)
        zlogvar = self.zlogvar(h2)
        z = self.gaussian_sample(z_mean, zlogvar)
        return z, z_mean, zlogvar


class Px_z(nn.Module):
    def __init__(self, decoder, k):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.decoder_hidden = self.decoder.hidden_size
        self.latent_dim = self.decoder.latent_dim
        self.z_mean = nn.Linear(k, self.latent_dim)
        self.zlogvar = nn.Linear(k, self.latent_dim)

    def forward(self, z, y):
        # p(z|y)
        z_mean = self.z_mean(y)
        zlogvar = self.zlogvar(y)

        # p(x|z)
        x_hat = self.decoder(z)
        return z_mean, zlogvar, x_hat


class EncoderFC(nn.Module):
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
        h = self.enc_block(x)
        return h


class DecoderFC(nn.Module):
    def __init__(self, input_size, hidden_size,
                 latent_dim, return_probs=False):
        """
        Decoder Module.
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
        out = self.dec_block(z)
        if self.return_probs:
            out = self.sigmoid(out)
        return out
