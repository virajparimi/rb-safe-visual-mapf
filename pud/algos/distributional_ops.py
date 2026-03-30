"""
Implementation of distributional operators with bug fix
"""

import torch


class CategoricalActivation:
    def __init__(self, vmin, vmax, N):
        self.vmin = vmin
        self.vmax = vmax
        self.N = N
        self.dz = (vmax - vmin) / (self.N - 1)
        self.zs = torch.linspace(0, self.vmax, self.N)

    def forward(self, probs: torch.Tensor, new_zs: torch.Tensor):
        new_zs = torch.clamp(new_zs, min=self.vmin, max=self.vmax)
        new_probs = torch.zeros_like(new_zs).to(new_zs.device)

        bs = (new_zs - self.vmin) / self.dz
        ls = torch.floor(bs).long()
        us = torch.ceil(bs).long()

        # When lower and upper adjacent class are the same, bump up the upper bound artificially by 1 to avoid
        # dumping the probabilities
        out_of_range_fix = (us == ls).to(torch.float)
        new_probs.scatter_add_(
            dim=-1, index=ls, src=probs * (us + out_of_range_fix - bs).to(probs.dtype)
        )
        new_probs.scatter_add_(dim=-1, index=us, src=probs * (bs - ls).to(probs.dtype))
        return new_probs


class TorchCategoricalVar:
    r"""
    We model the value distribution using a discrete distribution parametrized by
    $$N\in\mathbb{N}$$
    and
    $$V_{min},V_{max}\in\mathbb{R}$$,
    and whose support is the set of atoms
    $$\left\{ z_{i}=V_{min}+i\Delta z:0\le i<N\right\}$$
    $$\Delta z\doteq\frac{V_{max}-V_{min}}{N-1}$$

    Note that the class support $$z_i$$ is set in place and does not change

    The formal definition:
    $$
    Z_{\theta}\left(x,a\right)=z_{i}\;w.p.\;p_{i}\left(x,a\right)\doteq\frac{e^{\theta_{i}\left(x,a\right)}}{\sum_{j}e^{\theta_{j}\left(x,a\right)}}
    $$

    Meaning the probability of the variable being equal to $$z_i$$ is $$p_{i}\left(x,a\right)$$, which is a softmax

    ! variables of this class is NOT allowed to perform arithmetic operations with another variables of this same class
    """

    def __init__(self, Vmin: float, Vmax: float, probs: torch.Tensor):
        # Calculate the dimension of the categorical var
        if len(probs.shape) == 1:  # Without batch dim
            self.N = len(probs)
        elif len(probs.shape) == 2:  # batch, val
            self.N = len(probs[0])

        self.probs = probs

        self.dz = (Vmax - Vmin) / (self.N - 1)
        self.zs = torch.zeros(self.N)
        for i in range(self.N):
            self.zs[i] = Vmin + i * self.dz

        self.Vmin, self.Vmax = Vmin, Vmax

    def project(self, new_zs):
        """
        Faithfully follow the Algorithm 1 Categorical Algorithm, seems correct
        """
        new_zs = torch.clamp(new_zs, min=self.Vmin, max=self.Vmax)
        new_probs = torch.zeros_like(new_zs)

        bs = (new_zs - self.Vmin) / self.dz
        ls = torch.floor(bs).long()
        us = torch.ceil(bs).long()

        # When lower and upper adjacent class are the same, bump up the upper bound artificially by 1 to avoid
        # dumping the probabilities
        out_of_range_fix = (us == ls).to(torch.float)
        new_probs.scatter_add_(
            dim=-1,
            index=ls,
            src=self.probs * (us + out_of_range_fix - bs).to(self.probs.dtype),
        )
        new_probs.scatter_add_(
            dim=-1, index=us, src=self.probs * (bs - ls).to(self.probs.dtype)
        )
        return TorchCategoricalVar(self.Vmin, self.Vmax, new_probs)

    def mean(self):
        return torch.sum(self.probs * self.zs, dim=-1)

    def to(self, device='cpu'):
        device = torch.device(device)
        self.probs = self.probs.to(device)
        self.zs = self.zs.to(device)
