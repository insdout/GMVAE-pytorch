import torch
from torch import nn
from modules import Qy_x, Qz_xy, Px_z, EncoderFC, DecoderFC

"""
link:
https://github.com/sghalebikesabi/gmm-vae-clustering-pytorch/blob/master/gmmvae.py

def forward(self, x):
        xb = x
        y_ = torch.zeros([x.shape[0], 10])
        qy_logit, qy = self.qy_graph(xb)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * 10 for i in range(6)]
        for i in range(10):
            y = y_ + torch.eye(10)[i]
            z[i], zm[i], zv[i] = self.qz_graph(xb, y)
            zm_prior[i], zv_prior[i], px[i] = self.decoder(z[i], y)
        
        latent_samples = {'z': z}
        variational_params = {
            'zm': zm,
            'zv': zv, 
            'zm_prior': zm_prior, 
            'zv_prior': zv_prior,
            'qy_logit': qy_logit,
            'qy': qy,
        }

        return px, variational_params, latent_samples
"""

class GMVAE(nn.Module):
    def __init__(self, Qy_x, Qz_xy, Px_z, k):
        super(GMVAE, self).__init__()
        self.k = k
        self.qy_x = Qy_x
        self.qz_xy = Qz_xy
        self.px_z = Px_z
    
    def forward(self, x):
        k = self.k
        batch_size = x.shape[0]
        y_ = torch.zeros([batch_size, k])
        qy_logit, qy = self.qy_x(x)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * k for i in range(6)]
        for i in range(k):
            y = y_ + torch.eye(k)
            z[i], zm[i], zv[i] = self.qz_xy(x, y)
            zm_prior[i], zv_prior[i], px[i] = self.px_z(z[i], y)
        x_hat = torch.sum(qy*px)

        out = {
            "z": z,
            "zm": zm,
            "zv": zv,
            "zm_prior": zm_prior,
            "zv_prior": zv_prior,
            "qy_logit": qy_logit,
            "qy": qy,
            "px": px,
            "x_hat": x_hat}
        return out

