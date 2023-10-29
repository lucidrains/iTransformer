import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import Optional, Union, Tuple

# to understand Alex's brilliant (un)pack abstraction, please go through the following tutorial
# https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from iTransformer.attend import Attend
from iTransformer.revin import RevIN

from rotary_embedding_torch import RotaryEmbedding

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t):
    return (t,) if not isinstance(t, tuple) else t

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 4,
        dropout = 0.,
        causal = False,
        flash = True,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, dim_inner, bias = False),
            nn.SiLU(),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        self.attend = Attend(flash = flash, dropout = dropout, causal = causal)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    @beartype
    def forward(self, x):
        q, k, v = self.to_qkv(x)

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        out = self.attend(q, k, v)

        out = out * self.to_v_gates(x)
        return self.to_out(out)

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = rearrange(x, '... (r d) -> r ... d', r = 2)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

# transformer block

class TransformerBlock(Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        flash_attn = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb

        self.attn = Attention(flash = flash_attn, rotary_emb = rotary_emb, causal = causal, dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x, rotary_emb: Optional[RotaryEmbedding] = None):

        x = self.attn(x) + x
        x = self.attn_norm(x)

        x = self.ff(x) + x
        x = self.ff_norm(x)

        return x

# main class

class iTransformer2D(Module):
    @beartype
    def __init__(
        self,
        *,
        num_variates: int,
        lookback_len: int,
        num_time_tokens: int,
        depth: int,
        dim: int,
        pred_length: Union[int, Tuple[int, ...]],
        dim_head = 32,
        heads = 4,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        num_mem_tokens = 4,
        use_reversible_instance_norm = False,
        flash_attn = True
    ):
        super().__init__()
        assert divisible_by(lookback_len, num_time_tokens)
        assert num_time_tokens >= 2

        self.num_variates = num_variates
        self.lookback_len = lookback_len
        self.num_time_tokens = num_time_tokens

        self.mem_tokens = nn.Parameter(torch.randn(num_mem_tokens, dim)) if num_mem_tokens > 0 else None

        pred_length = cast_tuple(pred_length)
        self.pred_length = pred_length

        self.reversible_instance_norm = RevIN(num_variates) if use_reversible_instance_norm else None

        rotary_emb = RotaryEmbedding(dim_head)

        self.layers = ModuleList([])

        block_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            ff_mult = ff_mult,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn
        )

        for _ in range(depth):
            self.layers.append(ModuleList([
                TransformerBlock(causal = True, rotary_emb = rotary_emb, **block_kwargs),
                TransformerBlock(causal = False, **block_kwargs)
            ]))

        self.to_variate_token = nn.Sequential(
            nn.Linear(lookback_len, dim),
            nn.LayerNorm(dim)
        )

        time_kernel_size = lookback_len // num_time_tokens

        self.to_time_tokens = nn.Sequential(
            Rearrange('b v n -> (b v) 1 n'),
            nn.ConstantPad1d((time_kernel_size, 0), value = 0.),
            nn.Conv1d(1, dim, time_kernel_size * 2),
            Rearrange('(b v) d t -> b v t d', v = num_variates),
            nn.LayerNorm(dim)
        )

        self.pred_heads = ModuleList([])

        for one_pred_length in pred_length:
            head = nn.Sequential(
                nn.Linear(dim, one_pred_length),
                Rearrange('b v n -> b n v')
            )

            self.pred_heads.append(head)

    @beartype
    def forward(
        self,
        x: Tensor,
        targets: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ):
        """
        einstein notation

        b - batch
        n - time
        v - variate
        t - number of time tokens
        """

        has_mem = exists(self.mem_tokens)
        assert x.shape[1:] == (self.lookback_len, self.num_variates)

        # the crux of the paper is basically treating variates as the spatial dimension in attention
        # there is a lot of opportunity to improve on this, if the paper is successfully replicated

        x = rearrange(x, 'b n v -> b v n')

        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        # derive the time tokens per variate 't'

        t = self.to_time_tokens(x)

        # 'v' will be the variate pool token, which is the same as the token per variate from iTransformer

        v = self.to_variate_token(x)

        # combine time and variate tokens into 2d feature map of variates and time

        x, variate_pool_token_ps = pack((t, v), 'b v * d')

        # memory tokens

        if has_mem:
            m = repeat(self.mem_tokens, 'm d -> b m t d', b = x.shape[0], t = x.shape[-2])
            x, mem_ps = pack([m, x], 'b * t d')

        # attention and feedforward layers

        for time_attn_block, variate_attn_block in self.layers:
            x, ps = pack_one(x, '* t d')

            # causal attention across time for each variate
            x = time_attn_block(x)

            x = unpack_one(x, ps, '* t d')

            x = rearrange(x, 'b v t d -> b t v d')
            x, ps = pack_one(x, '* v d')

            # full attention across variates (as in inverted Transformer paper)
            x = variate_attn_block(x)

            x = unpack_one(x, ps, '* v d')
            x = rearrange(x, 'b t v d -> b v t d')

        # splice out memory tokens

        if has_mem:
            _, x = unpack(x, mem_ps, 'b * t d')

        # get back the original variate pooled tokens

        _, v = unpack(x, variate_pool_token_ps, 'b v * d')

        # reversible instance normaization, if needed

        if exists(self.reversible_instance_norm):
            v = reverse_fn(v)

        # predicting multiple times

        pred_list = [fn(v) for fn in self.pred_heads]

        # calculate loss if targets is passed in

        if exists(targets):
            targets = cast_tuple(targets)
            assert len(targets) == len(pred_list)

            assert self.training
            mse_loss = 0.
            for target, pred in zip(targets, pred_list):
                assert targets.shape == pred_list.shape

                mse_loss = mse_loss + F.mse_loss(target, pred)

            return mse_loss

        if len(pred_list) == 0:
            return pred_list[0]

        pred_dict = dict(zip(self.pred_length, pred_list))
        return pred_dict
