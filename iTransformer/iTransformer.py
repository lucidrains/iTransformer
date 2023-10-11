import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

class iTransformer(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
