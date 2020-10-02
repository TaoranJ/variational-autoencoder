import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

import flow


# =============================================================================
# ================================= Encoder ===================================
# =============================================================================
class VariationalMeanField(nn.Module):
    """Approximate posterior parameterized by an inference network. Encoder.
    x -> z

    Parameters
    ----------
    data_dim : int
        Dimension of the input data x.
    latent_dim : int
        Dimension of the latent space z.

    """

    def __init__(self, data_dim, latent_dim):
        super().__init__()
        self.inference_network = FNN(data_dim, latent_dim * 2, latent_dim * 2)
        self.log_q_z = LogGaussianProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""

        mu, sigma_arg = torch.chunk(self.inference_network(x).unsqueeze(1),
                                    chunks=2, dim=-1)
        sigma = self.softplus(sigma_arg)
        # Sample z. Reparameterization (only for Gaussian distribution).
        std_gaussian = torch.randn((mu.shape[0], n_samples, mu.shape[-1]),
                                   device=mu.device)
        z = mu + sigma * std_gaussian
        # Variational prob (N, 1, 1)
        log_q_z = self.log_q_z(mu, sigma, z).sum(-1, keepdim=True)
        return z, log_q_z


class VariationalFlow(nn.Module):
    """Approximate posterior parameterized by a flow
    (https://arxiv.org/abs/1606.04934)."""
    def __init__(self, latent_dim, data_dim, flow_depth):
        super().__init__()
        hidden_dim = latent_dim * 2
        self.inference_network = FNN(data_dim, hidden_dim, latent_dim * 3)
        modules = []
        for _ in range(flow_depth):
            modules.append(flow.InverseAutoregressiveFlow(
                num_input=latent_dim, num_hidden=hidden_dim,
                num_context=latent_dim))
            modules.append(flow.Reverse(latent_dim))
        self.q_z_flow = flow.FlowSequential(*modules)
        self.log_q_z_0 = LogGaussianProb()
        self.softplus = nn.Softplus()

    def forward(self, x, n_samples=1):
        """Return sample of latent variable and log prob."""
        loc, scale_arg, h = torch.chunk(self.inference_network(x).unsqueeze(1),
                                        chunks=3, dim=-1)
        scale = self.softplus(scale_arg)
        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]),
                          device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc, scale, z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow).sum(-1, keepdim=True)
        return z_T, log_q_z


# =============================================================================
# ================================= Decoder ===================================
# =============================================================================
class Model(nn.Module):
    """Bernoulli model parameterized by a generative network with Gaussian
    latents for MNIST."""
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.register_buffer('p_z_mu', torch.zeros(latent_dim))
        self.register_buffer('p_z_sigma', torch.ones(latent_dim))
        self.log_p_z = LogGaussianProb()
        self.log_p_x = LogBernoulliProb()
        self.generative_network = FNN(latent_dim, latent_dim * 2, data_dim)

    def forward(self, z, x):
        """Return log probability of model."""
        log_p_z = self.log_p_z(self.p_z_mu, self.p_z_sigma, z).sum(
                -1, keepdim=True)
        logits = self.generative_network(z)
        # unsqueeze sample dimension
        logits, x = torch.broadcast_tensors(logits, x.unsqueeze(1))
        log_p_x = self.log_p_x(logits, x).sum(-1, keepdim=True)
        return log_p_z + log_p_x


# =============================================================================
# ================================== Utils ====================================
# =============================================================================
class FNN(nn.Module):
    """Three consecutive fully connected layers for approximation.

    input_dim -> hidden_dim -> hidden_dim -> output_dim.

    Parameters
    ----------
    input_dim : int
        Dimension of the input.
    hidden_dim : int
        Dimension of the inner state.
    output_dim : int
        Dimension of the output.

    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim))

    def forward(self, input):
        return self.net(input)


class LogGaussianProb(nn.Module):
    """Logarithm of Gaussian distribution."""

    def __init__(self):
        super().__init__()

    def forward(self, mu, sigma, z):
        """Return the logrithm of Gaussian density function at point z.

        Parameters
        ----------
        mu : float
            Mean.
        sigma : float
            sigma^2 is the variance.

        Returns
        -------
        Logrithm of Gaussian density function at point z

        """

        sigma2 = torch.pow(sigma, 2)
        return -0.5 * torch.log(2 * np.pi * sigma2) \
            - torch.pow(z - mu, 2) / (2 * sigma2)


class LogBernoulliProb(nn.Module):
    """Logarithm of p^k * (1 - p)^(1 - k).

    log{p^k * (1 - p)^k} = klog(p) + (1 - k)log(1 - p), k in {0, 1}
    BCELoss = - ylog(x) + (1 - y)log(1 - x)
    Let y = k = targets, x = p in [0, 1], we can use BCEloss.

    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """"""
        return -F.binary_cross_entropy_with_logits(input, target,
                                                   reduction='none')
