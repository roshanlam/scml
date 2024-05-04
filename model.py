import torch
from torch import nn, Tensor
import torch.nn.functional as F
from functools import wraps
from dataclasses import dataclass
from typing import List, Dict
from math import log, ceil


class Intermediates:
    def __init__(
        self, qk_similarities=None, pre_softmax_attn=None, post_softmax_attn=None
    ):
        self.qk_similarities = qk_similarities
        self.pre_softmax_attn = pre_softmax_attn
        self.post_softmax_attn = post_softmax_attn


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        causal=False,
        heads=None,
        scale=None,
        qk_norm=False,
        flash=False,
        device=None,
    ):
        super().__init__()
        self.scale = scale or (1 / torch.sqrt(torch.tensor(heads, dtype=torch.float32)))
        self.causal = causal
        self.attn_fn = (
            partial(F.softmax, dtype=torch.float32) if not qk_norm else F.softmax
        )
        self.dropout = nn.Dropout(dropout)
        self.flash = flash
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        if heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)
        else:
            self.pre_softmax_talking_heads = self.post_softmax_talking_heads = None

    def forward(self, q, k, v, mask=None, attn_bias=None):
        scale = self.scale / torch.sqrt(q.size(-1))

        if self.pre_softmax_talking_heads:
            q = self.pre_softmax_talking_heads(q)
            k = self.pre_softmax_talking_heads(k)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_bias is not None:
            attn_weights += attn_bias

        if mask is not None:
            attn_weights.masked_fill_(mask, float("-inf"))

        if self.causal:
            i, j = q.size(-2), k.size(-2)
            causal_mask = torch.triu(
                torch.ones((i, j), device=self.device), diagonal=1
            ).bool()
            attn_weights.masked_fill_(causal_mask, float("-inf"))

        attn_probs = self.attn_fn(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        if self.post_softmax_talking_heads:
            attn_probs = self.post_softmax_talking_heads(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        return attn_output, Intermediates(
            qk_similarities=attn_weights, pre_softmax_attn=attn_probs
        )


def exists(val):
    return val is not None


def max_neg_value(dtype):
    return -torch.finfo(dtype).max


def eval_decorator(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        original_mode = self.training
        self.eval()
        result = fn(self, *args, **kwargs)
        self.train(original_mode)
        return result

    return wrapper


def top_k(logits, k):
    if k < logits.size(-1):
        values, indices = torch.topk(logits, k)
        mask = torch.ones_like(logits).scatter_(1, indices, 0)
        logits[mask.bool()] = float("-inf")
    return logits


def top_p(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, pad_value=0, ignore_index=-100, mask_prob=0):
        super().__init__()
        assert 0 <= mask_prob < 1, "mask_prob should be in the range [0, 1)"
        self.net = net
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.mask_prob = mask_prob

    @torch.no_grad()
    @eval_decorator
    def generate(
        self, start_tokens, seq_len, eos_token=None, temperature=1.0, top_k=0, top_p=1.0
    ):
        out = start_tokens
        for _ in range(seq_len):
            logits = self.net(out)[-1, :]
            if top_k > 0:
                logits = top_k(logits, top_k)
            if top_p < 1.0:
                logits = top_p(logits, top_p)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            out = torch.cat([out, next_token], dim=1)
            if eos_token and (next_token.item() == eos_token):
                break
        return out

    def forward(self, x):
        if self.mask_prob > 0:
            mask = torch.rand(x.shape[:-1], device=x.device) < self.mask_prob
            x = torch.where(mask, torch.tensor(self.pad_value, device=x.device), x)
        logits = self.net(x)
        loss = F.cross_entropy(
            logits.transpose(1, 2), x, ignore_index=self.ignore_index
        )
        return logits, loss


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x).pow(2)


class BaseTokenizer:
    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError(
            "Tokenize method needs to be implemented by subclass."
        )


class CustomTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> List[int]:
        return [ord(char) for char in text]


class BaseEmbedding:
    def get_embedding(self, num_tokens: int, dim: int) -> nn.Module:
        raise NotImplementedError(
            "Embedding method needs to be implemented by subclass."
        )


class SimpleEmbedding(BaseEmbedding):
    def get_embedding(self, num_tokens: int, dim: int) -> nn.Module:
        return nn.Embedding(num_tokens, dim)


class TokenEmbedding(nn.Module):
    def __init__(self, num_tokens, dim, embedding_provider: BaseEmbedding):
        super().__init__()
        self.embedding = embedding_provider.get_embedding(num_tokens, dim)

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_length, dim)
        self.scale = dim**-0.5

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        return self.positional_embeddings(positions) * self.scale


class DynamicPositionBias(nn.Module):
    def __init__(self, dim, heads, depth=2, log_distance=True):
        super().__init__()
        self.log_distance = log_distance
        self.linears = nn.ModuleList(
            [nn.Linear(1, dim) if i == 0 else nn.Linear(dim, dim) for i in range(depth)]
        )
        self.final_linear = nn.Linear(dim, heads)
        self.activation = nn.SiLU()

    def forward(self, i, j):
        distance = torch.abs(i - j).float().unsqueeze(-1)
        if self.log_distance:
            distance = torch.log1p(distance)
        for linear in self.linears:
            distance = self.activation(linear(distance))
        return self.activation(self.final_linear(distance))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", self.inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        freqs = torch.einsum(
            "i , j -> i j", torch.arange(seq_len, device=x.device), self.inv_freq
        )
        emb = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=-1)
        return emb


class LearnedAlibiPositionalBias(nn.Module):
    def __init__(self, num_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.heads = num_heads
        self.num_buckets = num_buckets
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def forward(self, positions):
        distance = positions[:, :, None] - positions[:, None, :]
        distance = distance.clamp(min=-max_distance, max=max_distance) + max_distance
        rp_bucket = distance // (2 * max_distance / self.num_buckets)
        return self.scale * self.relative_attention_bias(rp_bucket.to(torch.long))


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * (dim**-0.5))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm * self.scale


class Residual(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x, residual):
        return x + residual * self.scale


# GRU Gating Mechanism
class GRUGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, state):
        return self.gru(x.view(-1, x.size(-1)), state.view(-1, state.size(-1))).view_as(
            x
        )


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    amount = min(amount, t.shape[1])
    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0)
    return pad(t, (amount, -amount), "constant", 0)


class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = shifts

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask")
        splitted = x.split(x.shape[-1] // len(self.shifts), dim=-1)
        shifted = [shift(seg, amt, mask) for seg, amt in zip(splitted, self.shifts)]
        x = torch.cat(shifted, dim=-1)
        return self.fn(x, **kwargs)


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.GELU):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.activation = activation()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, activation=nn.GELU, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim_head * heads)
        self.scale = (dim_head) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, -1).permute(2, 0, 3, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if exists(mask):
            dots = dots.masked_fill(mask[:, None, None, :], float("-inf"))
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.out(out.reshape(b, n, -1))
