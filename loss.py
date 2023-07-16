import torch
from torch import nn
import numpy as np
import math
import logging
import torch.nn.functional as F


"""
useful links:
https://lucasdavid.github.io/blog/machine-learning/crossentropy-and-logits/

def loss_function(data, targets, px_logit, variational_params, latent_samples):

    nent = torch.sum(
        variational_params['qy'] *
        torch.nn.LogSoftmax(1)(variational_params['qy_logit']), 1)  ###

    losses = [None]*10
    for i in range(10):
        losses[i] = labeled_loss(
            data, px_logit[i],
            latent_samples['z'][i],
            variational_params['zm'][i],
            torch.exp(variational_params['zv'][i]),
            variational_params['zm_prior'][i],
            torch.exp(variational_params['zv_prior'][i]))

    loss = torch.stack([nent] + [variational_params['qy'][:, i] * losses[i]
    for i in range(10)]).sum(0)

    loss_dict = {
        'nent': nent.sum(),
        'optimization_loss': loss.sum(),
    }

    return loss_dict

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = torch.clamp(logits, -max_val, max_val)
    return -torch.sum(
        F.binary_cross_entropy(logits, x, reduction="none"), axis)

def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = torch.add(var, eps, name='clipped_var')
    return -0.5 * torch.sum(
        np.log(2 * math.pi) + torch.log(var) + torch.square(x - mu) / var, axis)

def test_acc(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        data = test_loader.dataset.data.view(-1, 784).to(device)/255.0
        labels = test_loader.dataset.targets.to(device)
        qy_logit, _ = model.qy_graph(data)
        cat_pred = qy_logit.argmax(1)
        real_pred = np.zeros_like(cat_pred)
        for cat in range(qy_logit.shape[1]):
            idx = cat_pred == cat
            lab = labels[idx]
            if len(lab) == 0:
                continue
            real_pred[cat_pred == cat] = lab.mode()[0]
    acc = np.mean(real_pred == test_loader.dataset.targets.numpy())
    return(acc)
"""


class MSE:
    """
    ADD DOCSTRING!!!
    """
    def __call__(self, x, x_hat):
        batch_size = x.shape[0]
        loss = nn.MSELoss(reduction='none')(x, x_hat)
        loss = loss.view(batch_size, -1).sum(axis=1)
        loss = loss.mean()
        return loss


class BCELogits:
    """
    ADD DOCSTRING!!!
    """
    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, x, px_logits):
        batch_size = x.shape[0]
        eps = self.eps
        if eps > 0.0:
            max_val = np.log(1.0 - eps) - np.log(eps)
            px_logits = torch.clamp(px_logits, -max_val, max_val)
        #loss = nn.BCEWithLogitsLoss(reduction="none")(px_logits, x)
        loss = -nn.BCELoss(reduction="none")(px_logits, x)
        loss = loss.view(batch_size, -1).sum(axis=1)
        return loss


class TotalLoss:
    """
    generative process:
    pθ(x, y, z) = pθ(x|z) pθ(z|y) p(y)
    y ~ Cat(y|1/k)
    z|y ~ N(z|µzθ(y), σ^2zθ(y))
    x|z ~ B(x|µxθ(z))

    The goal of GMVAE is to estimate the posterior
    distribution p(z, y|x),
    which is usually difficult to compute directly.
    Instead, a factorized posterior,
    known as the inference model,
    is commonly used as an approximation:

    qφ(z, y|x) = qφ(z|x, y) qφ(y|x)
    y|x ~ Cat(y|πφ(x))
    z|x, y ~ N(z|µzφ(x, y), σ^2zφ(x, y))

    ELBO = -KL(qφ(z|x, y) || pθ(z|y))
            - KL(qφ(y|x) || p(y)) + Eqφ(z|x,y) [log pθ(x|z)]

    """
    def __init__(self, k, recon_loss=MSE()):
        self.k = k
        self.recon_loss = recon_loss

    def negative_entropy_from_logit(self, qy, qy_logit):
        """
        Computes:
        ???
        ++++++++++++++++++++++++++++++++++++++++++++
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
        #return -0.5 * torch.sum(np.log(2 * math.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, axis)
        return -0.5 * torch.sum(
            np.log(2 * math.pi) + torch.log(var) + torch.square(x - mu) / var, axis)

    def _loss_per_class(self, x, x_hat, z, zm, zv, zm_prior, zv_prior):
        loss_px_i = -self.recon_loss(x, x_hat)
        print(f"px loss: {loss_px_i.sum().item()}")
        loss_px_i += self.log_normal(z, zm, zv) - self.log_normal(z, zm_prior, zv_prior)
        return loss_px_i - np.log(0.1)


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
                    x, px[i], z[i], zm[i], zv[i], zm_prior[i], zv_prior[i])
                )
            print(f"i: {i} my loss: {losses_i[-1]}")
        print([torch.sum(qy[:, i] * losses_i[i]).item() for i in range(self.k)])
        loss = torch.stack([loss_qy] + [qy[:, i] * losses_i[i] for i in range(self.k)]).sum(0)
        print(loss)
        # Alternative way to calculate loss:
        # loss_a = loss_qy +
        # torch.sum(torch.mul(torch.stack(losses_i), torch.transpose(qy, 1, 0)), dim=0)
        out_dict = {"nent": loss_qy.sum(), "optimization_loss": loss.sum()}
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
