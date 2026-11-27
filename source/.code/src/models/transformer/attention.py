"""Multi-head self-attention and the pre-norm transformer block."""
import math

import torch


class SelfAttention(torch.nn.Module):
    """Multi-head self-attention, optionally causal.

    ``qkv = Linear(dim, 3 * dim)`` makes queries, keys and values; each of the
    ``n_heads`` heads uses ``dim // n_heads`` of their channels:

        attention = softmax(Q K^T / sqrt(dim // n_heads)),  output = attention V

    With ``causal=True`` a token may not look at later tokens: the scores
    above the diagonal are set to ``-inf`` before the softmax. The heads are
    concatenated back and mixed by ``proj = Linear(dim, dim)``.
    ``self.attention`` keeps the last attention ``[batch, heads, T, T]``.
    """

    def __init__(self, dim, n_heads, causal=False):
        super().__init__()
        self.n_heads = n_heads
        self.causal = causal
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)
        self.attention = None

    def forward(self, signal):
        batch_size, n_tokens, dim = signal.shape
        head_dim = dim // self.n_heads
        # [batch, T, 3 * dim] -> 3 x [batch, heads, T, head_dim]
        query, key, value = (self.qkv(signal)
                             .reshape(batch_size, n_tokens, 3, self.n_heads, head_dim)
                             .permute(2, 0, 3, 1, 4))
        output = value
        ## Compute the attention of every head (masked when causal), store it
        ## in self.attention and apply it to the values
        ## YOUR CODE HERE
        # -- start --
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
        if self.causal:
            future = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=signal.device).triu(1)
            scores = scores.masked_fill(future, float('-inf'))
        self.attention = scores.softmax(dim=-1)
        output = self.attention @ value
        # -- end --
        # [batch, heads, T, head_dim] -> [batch, T, dim]
        output = output.transpose(1, 2).reshape(batch_size, n_tokens, dim)
        return self.proj(output)


class TransformerBlock(torch.nn.Module):
    """Pre-norm transformer block.

        x = x + attention(norm1(x))
        x = x + mlp(norm2(x)),  mlp = Linear(dim, 4 dim) -> GELU -> Linear(4 dim, dim)

    ``norm1`` and ``norm2`` are ``torch.nn.LayerNorm(dim)``.
    """

    def __init__(self, dim, n_heads, causal=False):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, n_heads, causal=causal)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim), torch.nn.GELU(), torch.nn.Linear(4 * dim, dim))

    def forward(self, signal):
        ## YOUR CODE HERE
        # -- start --
        signal = signal + self.attention(self.norm1(signal))
        signal = signal + self.mlp(self.norm2(signal))
        # -- end --
        return signal
