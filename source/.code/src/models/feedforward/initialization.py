"""Weight initialization of fully connected networks."""
import math

import torch


def initialize_(network, method, std=1.0, bound=1.0, generator=None):
    """Initialize every ``torch.nn.Linear`` of ``network`` in place.

    Biases become zero. Weights ``W [out_features, in_features]`` depend on
    ``method``, with ``fan_in = in_features``:

    - ``'zeros'``: ``W = 0``;
    - ``'normal'``: ``W ~ N(0, std^2)``;
    - ``'uniform'``: ``W ~ U(-bound, bound)``;
    - ``'he_normal'``: ``W ~ N(0, 2 / fan_in)``;
    - ``'he_uniform'``: ``W ~ U(-sqrt(6 / fan_in), sqrt(6 / fan_in))``, the
      same variance ``2 / fan_in`` as ``'he_normal'``.

    Parameters
    ----------
    network : torch.nn.Module
        The network; only its ``torch.nn.Linear`` layers change.
    method : str
        One of the methods above.
    std, bound : float
        Parameters of ``'normal'`` and ``'uniform'``.
    generator : torch.Generator, optional
        Source of the random numbers, for repeatable weights.
    """
    for layer in network.modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        fan_in = layer.in_features
        ## YOUR CODE HERE
        # -- start --
        with torch.no_grad():
            if layer.bias is not None:
                layer.bias.zero_()
            if method == 'zeros':
                layer.weight.zero_()
            elif method == 'normal':
                layer.weight.normal_(0.0, std, generator=generator)
            elif method == 'uniform':
                layer.weight.uniform_(-bound, bound, generator=generator)
            elif method == 'he_normal':
                layer.weight.normal_(0.0, math.sqrt(2.0 / fan_in), generator=generator)
            elif method == 'he_uniform':
                he_bound = math.sqrt(6.0 / fan_in)
                layer.weight.uniform_(-he_bound, he_bound, generator=generator)
            else:
                raise ValueError(f'unknown initialization method {method!r}')
        # -- end --


@torch.no_grad()
def activation_stds(network, signal):
    """Standard deviation of the output of every ``torch.nn.ReLU`` for one input batch."""
    stds = []
    hooks = [layer.register_forward_hook(lambda module, inputs, output: stds.append(float(output.std())))
             for layer in network.modules() if isinstance(layer, torch.nn.ReLU)]
    try:
        network({'data': {'image': signal}})
    finally:
        for hook in hooks:
            hook.remove()
    return stds
