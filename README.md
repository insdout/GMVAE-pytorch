# GMVAE-pytorch
Gaussian Mixture Variational Encoder

Sure! Here's the equation written in LaTeX format:

$$
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(y,z|x)} \left[ \log p_\theta(x|y, z) + \log p_\theta(y) + \log p(z) - \log q_\phi(y, z|x) \right] \\
= \sum_y q_\phi(y|x)(-L(x, y)) + H(q_\phi(y|x)) = -U(x)
$$

Please note that in LaTeX, \log is used for logarithm, \geq is used for greater than or equal to, \mathbb{E} is used for expectation, and \sum is used for summation.
