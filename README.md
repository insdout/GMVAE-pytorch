# GMVAE-pytorch
Gaussian Mixture Variational Encoder

$$ \log p_\theta(x) \geq \mathbb{E}{q_\phi (y,z|x)}\left[ \log p_\theta(x|y, z) + \log p_\theta(y) + \log p(z) - \log q_\phi(y, z|x) \right]$$
