"""A character-level GPT."""
import torch

from .attention import TransformerBlock


class GPT(torch.nn.Module):
    """Token and position embeddings, causal transformer blocks, a linear head.

    ``forward`` reads ``batch['data']['tokens']`` (``int64 [batch, T]``,
    ``T <= context``) and writes ``batch['signals']['output']``, the logits
    of the next token at every position (``[batch, T, vocab_size]``), and
    ``batch['postprocessed']['next_token']`` (their argmax).
    """

    def __init__(self, vocab_size, context=128, dim=128, n_layers=4, n_heads=4):
        super().__init__()
        self.context = context
        # Placeholder layers: the network runs, wrongly, until they are defined.
        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.position_embedding = torch.nn.Parameter(torch.zeros(context, dim))
        self.blocks = torch.nn.Identity()
        self.norm = torch.nn.Identity()
        self.head = torch.nn.Linear(dim, vocab_size)

        ## Define the position embedding (random, std 0.02), n_layers causal
        ## TransformerBlocks in a torch.nn.Sequential and the final LayerNorm
        ## YOUR CODE HERE
        # -- start --
        self.position_embedding = torch.nn.Parameter(torch.randn(context, dim) * 0.02)
        self.blocks = torch.nn.Sequential(
            *[TransformerBlock(dim, n_heads, causal=True) for _ in range(n_layers)])
        self.norm = torch.nn.LayerNorm(dim)
        # -- end --

    def forward(self, batch):
        tokens = batch['data']['tokens']
        signal = self.token_embedding(tokens) + self.position_embedding[:tokens.shape[1]]
        signal = self.head(self.norm(self.blocks(signal)))
        batch['signals'] = {'output': signal}
        batch['postprocessed'] = {'next_token': signal.argmax(dim=-1)}
        return batch


@torch.no_grad()
def generate(model, tokens, n_new_tokens, temperature=1.0, generator=None):
    """Continue ``tokens`` (``int64 [T]``) by ``n_new_tokens`` sampled tokens.

    At every step the model sees the last ``model.context`` tokens; the next
    token is sampled from ``softmax(logits / temperature)`` of the last
    position (``torch.multinomial`` with ``generator``). Returns
    ``int64 [T + n_new_tokens]``.
    """
    model.eval()
    tokens = tokens.clone()
    for _ in range(n_new_tokens):
        next_token = tokens[-1:]
        ## YOUR CODE HERE
        # -- start --
        logits = model({'data': {'tokens': tokens[None, -model.context:]}})['signals']['output']
        probabilities = (logits[0, -1] / temperature).softmax(dim=-1)
        next_token = torch.multinomial(probabilities, 1, generator=generator)
        # -- end --
        tokens = torch.cat([tokens, next_token])
    return tokens
