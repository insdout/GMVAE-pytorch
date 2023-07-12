import torch
from torch import nn 

"""
# vae subgraphs
def qy_graph(x, k=10):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qy')) > 0
    # -- q(y)
    with tf.variable_scope('qy'):
        h1 = Dense(x, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        qy_logit = Dense(h2, k, 'logit', reuse=reuse)
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy
"""

class Qy_x(nn.Module):
    def __init__(self, encoder, enc_out_dim, k):
        super(Qy_x, self).__init__()
        self.encoder = encoder
        self.qy_logit = nn.Linear(enc_out_dim, k)
        self.qy = nn.Softmax()
    
    def forward(self, x):
        h1 = self.encoder(x)
        qy_logit = self.qy_logit(h1)
        qy = self.qy(qy_logit)
        return qy_logit, qy

"""
# vae subgraphs
def qz_graph(x, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='qz')) > 0
    # -- q(z)
    with tf.variable_scope('qz'):
        xy = tf.concat(1, (x, y), name='xy/concat')
        h1 = Dense(xy, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        zm = Dense(h2, 64, 'zm', reuse=reuse)
        zv = Dense(h2, 64, 'zv', tf.nn.softplus, reuse=reuse)
        z = GaussianSample(zm, zv, 'z')
    return z, zm, zv
"""    

class Qz_xy(nn.Module):
    def __init__(self, encoder, enc_out_dim, y_dim, h2_dim, z_dim):
        super(Qz_xy, self).__init__()
        self.encoder = encoder
        self.h2 = nn.Sequential(
            nn.Linear(enc_out_dim + y_dim, h2_dim), 
            nn.ReLU()
            )
        self.zm = nn.Linear(h2_dim, z_dim)
        self.zv = nn.Sequential(nn.Linear(h2_dim, z_dim), nn.Softplus())
    
    def gaussian_sample(self, zm, zv):
        pass
    def forward(self, x, y):
        h1 = self.encoder(x)
        xy = torch.cat((h1, y), dim=1)
        h2 = self.h2(xy)
        zm = self.zm(h2)
        zv = self.zv(h2)
        z = self.gaussian_sample(zm, zv)
        return z, zm, zv


"""
link:
https://github.com/jariasf/GMVAE/blob/master/pytorch/networks/Networks.py

# Generative Network
class GenerativeNet(nn.Module):

    # p(z|y)
    self.y_mu = nn.Linear(y_dim, z_dim)
    self.y_var = nn.Linear(y_dim, z_dim)

    # p(x|z)
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, x_dim),
        torch.nn.Sigmoid()
    ])

    # p(z|y)
    ++++++++++++++++++++++++++++++++++++++++++
    +     Очень странные имена переменных!   +
    ++++++++++++++++++++++++++++++++++++++++++
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var
    
    # p(x|z)
    ++++++++++++++++++++++++++++++++++++++++++
    +     Очень странные имена переменных!   +
    ++++++++++++++++++++++++++++++++++++++++++
    def pxz(self, z):
        for layer in self.generative_pxz:
        z = layer(z)
        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)
        
        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output

===============================
link:
https://github.com/RuiShu/vae-clustering/blob/master/gmvae.py

def px_graph(z, y):
    reuse = len(tf.get_collection(tf.GraphKeys.VARIABLES, scope='px')) > 0
    # -- p(z)
    with tf.variable_scope('pz'):
        zm = Dense(y, 64, 'zm', reuse=reuse)
        zv = Dense(y, 64, 'zv', tf.nn.softplus, reuse=reuse)
    # -- p(x)
    with tf.variable_scope('px'):
        h1 = Dense(z, 512, 'layer1', tf.nn.relu, reuse=reuse)
        h2 = Dense(h1, 512, 'layer2', tf.nn.relu, reuse=reuse)
        px_logit = Dense(h2, 784, 'logit', reuse=reuse)
    return zm, zv, px_logit
===============================
"""
class Px_z(nn.Module):
    def __init__(self, decoder):
        super(Px_z, self).__init__()
        self.decoder = decoder
        self.zm = nn.Linear()
        self.zv = nn.Sequential(
            nn.Linear(),
            nn.Softplus()
            )
    
    def forward(self, z, y):
        # p(z)
        zm = self.zm(y)
        zv = self.zv(y)

        # p(x)
        x_hat = self.decoder(z)
        return zm, zv, x_hat
    
    
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, dropout_lstm, dropout=0, num_layers=1, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.p_lstm = dropout_lstm
        self.p = dropout

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.p_lstm,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.fc_mean = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * hidden_size,
                out_features=latent_dim)
        )

        self.fc_log_var = nn.Sequential(
            nn.Dropout(self.p),
            nn.Linear(
                in_features=self.num_directions * hidden_size,
                out_features=latent_dim)
        )

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

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
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        return z, mean, log_var


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
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