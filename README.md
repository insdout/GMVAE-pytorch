# Gaussian Mixture Variational Autoencoder (GMVAE)

This repository contains an implementation of the Gaussian Mixture Variational Autoencoder (GMVAE) based on the paper "A Note on Deep Variational Models for Unsupervised Clustering" by James Brofos, Rui Shu, and Curtis Langlotz and a modified version of the M2 model proposed by D. P. Kingma et al. in their paper "Semi-Supervised Learning with Deep Generative Models."

$$ \log p_\theta(x) \geq \mathbb{E}{q_\phi (y,z|x)}\left[ \log p_\theta(x|y, z) + \log p_\theta(y) + \log p(z) - \log q_\phi(y, z|x) \right]$$

<h2 align="center"> P(Y|X) over epochs during unsupervised training:</h2>
<p align="center">
<img alt="probability over epochs" src="./outputs/train.gif" width="600">
</p>

## Repository Structure

The repository has the following structure:

```
.
├── configs
│ ├── config.yaml
│ └── model
│ └── gmvae_fc.yaml
├── loss.py
├── models.py
├── modules.py
├── README.md
├── test.ipynb
├── train.py
└── utils.py
```

- `configs`: Contains the configuration files for the GMVAE model, including `config.yaml` for general settings and `gmvae_fc.yaml` for model-specific settings.
- `loss.py`: Implements the loss functions used in the GMVAE.
- `models.py`: Defines the GMVAE model architecture.
- `modules.py`: Contains custom modules used in the GMVAE.
- `train.py`: The main script for training the GMVAE model.
- `utils.py`: Contains utility functions used in the GMVAE implementation.

## Usage

To train and evaluate the GMVAE model, follow these steps:

1. Configure the model settings in `config.yaml` and `gmvae_fc.yaml` as needed.
2. Run the `train.py` script to train the GMVAE model.

Make sure to install the required dependencies before running the code. You may use a virtual environment and install the dependencies using the provided `requirements.txt` file.

## References


D. P. Kingma, D. J. Rezende, S. Mohamed, and M. Welling. Semi-Supervised Learning with Deep
Generative Models. ArXiv e-prints, June 2014.
