import torch
from torch import nn
import numpy as np
import math
import logging


class MSE:
    """
    ADD DOCSTRING!!!
    """
    def __call__(self, x, x_hat):
        batch_size = x.shape[0]
        loss = nn.MSELoss(reduction='none')(x, x_hat)
        loss = loss.view(batch_size, -1).sum(axis=1)
        return loss


class BCELogits:
    """
    ADD DOCSTRING!!!
    """
    def __init__(self, eps=0.0):
        self.eps = eps

    def __call__(self, x, px_logits):
        batch_size = x.shape[0]
        eps = self.eps
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            px_logits = torch.clamp(px_logits, -max_val, max_val)
        # loss = nn.BCEWithLogitsLoss(reduction="none")(px_logits, x)
        # we dont use logits: last layer is sigmois
        loss = nn.BCELoss(reduction="none")(px_logits, x)
        loss = loss.view(batch_size, -1).sum(axis=1)
        return loss


class TotalLoss:
    """
    generative process:
    p_theta(x, y, z) = p_theta(x|z) p_theta(z|y) p(y)
    y ~ Cat(y|1/k)
    z|y ~ N(z|mu_z_theta(y), sigma^2*z_theta(y))
    x|z ~ B(x|mu_x_theta(z))

    The goal of GMVAE is to estimate the posterior
    distribution p(z, y|x),
    which is usually difficult to compute directly.
    Instead, a factorized posterior,
    known as the inference model,
    is commonly used as an approximation:

    q_phi(z, y|x) = q_phi(z|x, y) q_phi(y|x)
    y|x ~ Cat(y|pi_phi(x))
    z|x, y ~ N(z|mu_z_phi(x, y), sigma^2z_phi(x, y))

    ELBO = -KL(q_phi(z|x, y) || p_theta(z|y))
            - KL(q_phi(y|x) || p(y)) + Eq_phi(z|x,y) [log p_theta(x|z)]

    """
    def __init__(self, k, recon_loss=MSE()):
        self.k = k
        self.recon_loss = recon_loss

    def negative_entropy_from_logit(self, qy, qy_logit):
        """
        Computes:
        ???
        ++++++++++++++++++++++++++++++++++++++++++++
        H(q, q) = - ∑q*log q
        H(q, q_logit) = - ∑q*log p(q_logit)
        p(q_logit) = softmax(qy_logit)
        H(q, q_logit) = - ∑q*log softmax(qy_logit)
        ++++++++++++++++++++++++++++++++++++++++++++
        """
        nent = torch.sum(qy * torch.nn.LogSoftmax(1)(qy_logit), 1)
        return nent

    def log_normal(self, x, mu, var, eps=0., axis=-1):
        """
        ADD DOCSTRING!!!
        """
        if eps > 0.0:
            var = torch.add(var, eps)
        return -0.5 * torch.sum(np.log(2 * math.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, axis)

    def _loss_per_class(self, x, x_hat, z, zm, zv, zm_prior, zv_prior):
        loss_px_i = self.recon_loss(x, x_hat)
        loss_px_i += self.log_normal(z, zm, zv) - self.log_normal(z, zm_prior, zv_prior)
        return loss_px_i - np.log(1/self.k)

    def __call__(self, x, output_dict):
        qy = output_dict["qy"]
        qy_logit = output_dict["qy_logit"]
        px = output_dict["px"]
        z = output_dict["z"]
        zm = output_dict["zm"]
        zv = output_dict["zv"]
        zm_prior = output_dict["zm_prior"]
        zv_prior = output_dict["zv_prior"]
        loss_qy = self.negative_entropy_from_logit(qy, qy_logit)
        losses_i = []
        for i in range(self.k):
            logging.debug(f"Class: {i}")
            losses_i.append(
                self._loss_per_class(
                    x, px[i], z[i], zm[i], torch.exp(zv[i]), zm_prior[i], torch.exp(zv_prior[i]))
                )
        loss = torch.stack([loss_qy] + [qy[:, i] * losses_i[i] for i in range(self.k)]).sum(0)
        # Alternative way to calculate loss:
        # torch.sum(torch.mul(torch.stack(losses_i), torch.transpose(qy, 1, 0)), dim=0)
        out_dict = {"cond_entropy": loss_qy.sum(), "total_loss": loss.sum()}
        return out_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    batch_size = 3
    input_dim = 4
    k = 5
    latent_dim = 2
    p = 0.5  # Probability of success (0 or 1)

    x = torch.bernoulli(torch.full((batch_size, input_dim, input_dim), p))
    x = x.reshape((batch_size, -1))
    # x_hat = torch.rand((k, batch_size, input_dim**2))
    x_hat = 2*torch.stack([x.reshape(batch_size, -1) for i in range(k)]) - 1
    px = x_hat
    qy_logit = 2*torch.rand((batch_size, k))
    qy = torch.nn.functional.softmax(qy_logit, dim=-1)
    logging.debug(f"initial qy: {qy}")
    logging.debug(f"initial qy sum: {qy.sum(1)}")
    z = torch.rand((k, batch_size, latent_dim))
    zm = zv = torch.rand((k, batch_size, latent_dim))
    zm_prior = zv_prior = torch.rand((k, batch_size, latent_dim))

    out = {
            "z": z,
            "zm": zm,
            "zv": zv,
            "zm_prior": zm_prior,
            "zv_prior": zv_prior,
            "qy_logit": qy_logit,
            "qy": qy,
            "px": px
            }

    loss = TotalLoss(k=k, recon_loss=MSE())
    res = loss(x=x.reshape(batch_size, -1), output_dict=out)
    print(f"Total Loss: {res}")

    loss = TotalLoss(k=k, recon_loss=BCELogits())
    res = loss(x=x.reshape(batch_size, -1), output_dict=out)
    print(f"Total Loss: {res}")
