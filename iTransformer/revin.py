import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import Tuple, Callable

# reversible instance normalization
# proposed in https://openreview.net/forum?id=cGDAkQo1C0p

class RevIN(Module):
    def __init__(self, num_variates, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1))
        self.beta = nn.Parameter(torch.zeros(num_variates, 1))

    @beartype
    def forward(self, x) -> Tuple[Tensor, Callable[Tensor, Tensor]]:
        assert x.shape[1] == self.num_variates

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        var_rsqrt = var.clamp(min = self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min = self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        return rescaled, reverse_fn
