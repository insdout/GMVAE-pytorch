import math
import numpy as np

import torch
import torch.nn.functional as F


def loss_function(data, targets, px_logit, variational_params, latent_samples):

    nent = torch.sum(variational_params['qy'] * torch.nn.LogSoftmax(1)(variational_params['qy_logit']), 1)  ###

    losses = [None]*10
    for i in range(10):
        losses[i] = labeled_loss(data, px_logit[i], latent_samples['z'][i], variational_params['zm'][i], torch.exp(variational_params['zv'][i]), variational_params['zm_prior'][i], torch.exp(variational_params['zv_prior'][i]))
        print(f"i: {i} original loss: {losses[i]}")
    print([torch.sum(variational_params['qy'][:, i] * losses[i]).item() for i in range(10)])
    loss = torch.stack([nent] + [variational_params['qy'][:, i] * losses[i] for i in range(10)]).sum(0)
    print(loss)
    loss_dict = {
        'nent': nent.sum(),
        'optimization_loss': loss.sum(),     
    }

    return loss_dict

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    print(f"px loss: {xy_loss.sum().item()}")
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


if __name__ == "__main__":
    from utils import get_model
    import torch.optim as optim
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt


    k = 10
    encoder_type = "FC"
    input_size = 28*28
    hidden_size = 128
    latent_dim = 32
    device = "cpu"
    model, my_loss_fn = get_model(k, encoder_type, input_size, hidden_size, latent_dim,
                                recon_loss_type="BCE", return_probs=True, eps=1e-8,
                                encoder_kwargs={}, decoder_kwargs={})

    model.to(device)
    model.train()
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=False)
    images, labels = next(iter(train_loader))
    images = images.reshape((-1, 28*28)).to(device)
    print(images.shape)
    #print(images.unique())
    out_train, out_infer = model(images)
    """
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
    ========================================
        latent_samples = {'z': z}
        variational_params = {
            'zm': zm,
            'zv': zv, 
            'zm_prior': zm_prior, 
            'zv_prior': zv_prior,
            'qy_logit': qy_logit,
            'qy': qy,
        }
    """
    px_logit = out_train["px"]
    latent_samples = {"z": out_train["z"]}
    variational_params = {
            'zm': out_train["zm"],
            'zv': out_train["zv"], 
            'zm_prior': out_train["zm_prior"], 
            'zv_prior': out_train["zv_prior"],
            'qy_logit': out_train["qy_logit"],
            'qy': out_train["qy"]
        }
    my_loss = my_loss_fn(images, out_train)
    orig_loss = loss_function(images, labels, px_logit, variational_params, latent_samples)

    my_nent = my_loss["nent"].item()
    my_full = my_loss["optimization_loss"].item()
    print(f"my loss nent: {my_nent} full: {my_full}")

    orig_nent = orig_loss["nent"].item()
    orig_full = orig_loss["optimization_loss"].item()
    print(f"orig loss nent: {orig_nent} full: {orig_full}")
