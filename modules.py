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
        h1 = self.h1(x)    # dim: Batch, hidden_size
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
    def __init__(self, input_size, hidden_size, latent_dim, return_probs=False):
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


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_lstm, num_layers=1, bidirectional=True):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = dropout_lstm

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )

    def forward(self, x):
        batch_size = x.shape[0]
        _, (h_n, _) = self.lstm(x)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        """
        hidden.shape = (num_layers*num_directions, batch, hidden_size)
        layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        So you shouldn’t simply do hidden[-1] but first do a view() to separate the num_layers and num_directions (1 or 2). If you do

        hidden = hidden.view(num_layers, 2, batch, hidden_size) # 2 for bidirectional
        last_hidden = hidden[-1]
        then last_hidden.shape = (2, batch, hidden_size) and you can do

        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]

        TODO: check if it same as
        # Pass the input through the LSTM
        output, (h_n, c_n) = lstm(input_data, (h0, c0))
        Extract the last forward and backward outputs
        last_forward_output = output[:, -1, :hidden_size]
        last_backward_output = output[:, 0, hidden_size:]

        """
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            h = torch.cat((h_n[-1, -2, :, :], h_n[-1, -1, :, :]), dim=1)
        else:
            h = h_n[-1, -1, :, :]
        return h


class DecoderLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 latent_dim,
                 window_size,
                 dropout_lstm,
                 dropout_layer,
                 num_layers=1,
                 bidirectional=True
                 ):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.window_size = window_size
        self.p_lstm = dropout_lstm
        self.p_dropout_layer = dropout_layer
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm_to_hidden = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.dropout_layer = nn.Dropout(self.p_dropout_layer)

        self.lstm_to_output = nn.LSTM(
            input_size=self.num_directions * hidden_size,
            hidden_size=input_size,
            batch_first=True
        )

    def forward(self, z):
        latent_z = z.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm_to_hidden(latent_z)
        out = self.dropout_layer(out)
        out, _ = self.lstm_to_output(out)
        return out
