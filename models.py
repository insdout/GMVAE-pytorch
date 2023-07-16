import torch
from torch import nn

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
            "x_hat": x_hat
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
