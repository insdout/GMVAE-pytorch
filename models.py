import torch
from torch import nn
import torch.nn.functional as F


class GMVAE(nn.Module):
    def __init__(self, k, Qy_x_net, Qz_xy_net, Px_z_net):
        super(GMVAE, self).__init__()
        self.k = k
        self.qy_x = Qy_x_net
        self.qz_xy = Qz_xy_net
        self.px_z = Px_z_net

    def infer(self, x):
        k = self.k
        batch_size = x.shape[0]
        qy_logit, qy = self.qy_x(x)
        y_hat = torch.argmax(qy, dim=-1)

        # Create tensor with 1s at specified indices
        y_ = torch.zeros(batch_size, k)
        y_ = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
        z_hat, *_ = self.qz_xy(x, y_)
        *_, x_hat = self.px_z(z_hat, y_)
        out_infer = {
            "y": y_hat,
            "z": z_hat,
            "x_hat": x_hat
            }
        return out_infer

    def forward(self, x):
        k = self.k
        batch_size = x.shape[0]
        y_ = torch.zeros([batch_size, k]).to(x.device)
        qy_logit, qy = self.qy_x(x)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * k for i in range(6)]
        for i in range(k):
            y = y_ + torch.eye(k).to(x.device)[i]
            z[i], zm[i], zv[i] = self.qz_xy(x, y)
            zm_prior[i], zv_prior[i], px[i] = self.px_z(z[i], y)
        
        # Inference for x_hat:
        with torch.no_grad():
            y_hat = torch.argmax(qy, dim=-1)
            y_temp = torch.zeros(batch_size, k)
            y_temp = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
            z_hat, *_ = self.qz_xy(x, y_temp)
            *_, x_hat = self.px_z(z_hat, y_temp)

        out_train = {
            "z": z,
            "zm": zm,
            "zv": zv,
            "zm_prior": zm_prior,
            "zv_prior": zv_prior,
            "qy_logit": qy_logit,
            "qy": qy,
            "px": px
            }

        out_infer = {
            "y": y_hat,
            "z": z_hat,
            "x_hat": x_hat,
            "qy": qy
            }
        return out_train, out_infer


class GMVAE2(torch.nn.Module):
    """VAE with GMM prior."""

    def __init__(self, input_size, k, latent_dim, hidden_size):
        super(GMVAE2, self).__init__()

        # input params
        self.input_dim = input_size
        self.r_cat_dim = k
        self.z_dim = latent_dim
        self.h_dim = hidden_size

        # q(y|x)
        self.fc_x_h = torch.nn.Linear(self.input_dim, self.h_dim)
        self.fc_hx_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_qyl = torch.nn.Linear(self.h_dim, self.r_cat_dim)
        self.fc_qyl_qy = torch.nn.Softmax(1)

        # q(z|x, y)
        self.fc_xy_h = torch.nn.Linear(self.input_dim + self.r_cat_dim, self.h_dim)
        self.fc_hxy_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_z = torch.nn.Linear(self.h_dim, self.z_dim*2)

        # p(z|y)
        self.fc_y_z = torch.nn.Linear(self.r_cat_dim, self.z_dim*2)

        # p(x|z)
        self.fc_z_h = torch.nn.Linear(self.z_dim, self.h_dim)
        self.fc_hz_h = torch.nn.Linear(self.h_dim, self.h_dim)
        self.fc_h_xl = torch.nn.Linear(self.h_dim, self.input_dim)

    def qy_graph(self, x):
        # q(y|x)
        hx = F.relu(self.fc_x_h(x))
        h = F.relu(self.fc_hx_h(hx))
        qy_logit = self.fc_h_qyl(h)
        qy = self.fc_qyl_qy(qy_logit)
        return qy_logit, qy

    def qz_graph(self, x, y):
        # q(z|x, y)
        xy = torch.cat([x, y], 1)

        hxy = F.relu(self.fc_xy_h(xy))
        h1 = F.relu(self.fc_hxy_h(hxy))
        z_post = self.fc_h_z(h1)
        z_mu_post, z_logvar_post = torch.split(z_post, self.z_dim, dim=1)
        z_std_post = torch.sqrt(torch.exp(z_logvar_post))

        eps = torch.randn_like(z_std_post)
        z = z_mu_post + eps*z_std_post

        return z, z_mu_post, z_logvar_post
  
    def decoder(self, z, y):

        # p(z)
        z_prior = self.fc_y_z(y)
        z_mu_prior, z_logvar_prior = torch.split(z_prior, self.z_dim, dim=1)

        # p(x|z)
        hz = F.relu(self.fc_z_h(z))
        h2 = F.relu(self.fc_hz_h(hz))
        x_logit = self.fc_h_xl(h2)

        return z_mu_prior, z_logvar_prior, torch.sigmoid(x_logit)

    def forward(self, x):
        xb = x
        batch_size = x.shape[0]
        k = self.r_cat_dim
        y_ = torch.zeros([x.shape[0], 10]).to(x.device)
        qy_logit, qy = self.qy_graph(xb)
        z, zm, zv, zm_prior, zv_prior, px = [[None] * 10 for i in range(6)]
        for i in range(10):
            y = y_ + torch.eye(10).to(x.device)[i]
            z[i], zm[i], zv[i] = self.qz_graph(xb, y)
            zm_prior[i], zv_prior[i], px[i] = self.decoder(z[i], y)

        with torch.no_grad():
            # Inference for x_hat:
            y_hat = torch.argmax(qy, dim=-1)
            y_temp = torch.zeros(batch_size, k)
            y_temp = torch.scatter(y_, 1, y_hat.unsqueeze(1), 1)
            z_hat, *_ = self.qz_graph(xb, y_temp)
            *_, x_hat = self.decoder(z_hat, y_temp)

        out_train = {
            "z": z,
            "zm": zm,
            "zv": zv,
            "zm_prior": zm_prior,
            "zv_prior": zv_prior,
            "qy_logit": qy_logit,
            "qy": qy,
            "px": px
            }

        out_infer = {
            "y": y_hat,
            "z": z_hat,
            "x_hat": x_hat,
            "qy": qy
            }
        return out_train, out_infer


if __name__ == "__main__":
    from utils import get_model

    k = 10
    encoder_type = "FC"
    input_size = 28*28
    hidden_size = 128
    latent_dim = 32

    model, criterion = get_model(k, encoder_type, input_size, hidden_size, latent_dim,
                                 recon_loss_type="BCE", return_probs=False, eps=0,
                                 encoder_kwargs={}, decoder_kwargs={})
    model.train()
    data = torch.randn((5, 28*28))
    out_train, out_infer = model(data)
    tr_x_hat = out_train["px"]
    inf_x_hat = out_infer["x_hat"]
    print(f"train x_hat shape: {torch.stack(tr_x_hat).shape}")
    print(f"infer x_hat shape: {inf_x_hat.shape}")

    print(criterion(data, out_train))

